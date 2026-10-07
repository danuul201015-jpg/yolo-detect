package com.qaz1sm.yolodetect

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Process
import android.os.SystemClock
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.text.Editable
import android.text.TextWatcher
import android.util.Range
import android.util.Size
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import com.qaz1sm.yolodetect.databinding.ActivityMainBinding
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private lateinit var b: ActivityMainBinding
    private lateinit var store: ModelStore
    private lateinit var ui: SettingsUi
    @Volatile private var detector: Detector? = null
    private val prefs by lazy { getSharedPreferences("app_prefs", MODE_PRIVATE) }

    // один поток под инференс, с повышенным приоритетом
    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread({
            try { Process.setThreadPriority(Process.THREAD_PRIORITY_DISPLAY) } catch (_: Throwable) {}
            r.run()
        }, "yolo-infer")
    }

    @Volatile private var cameraMode = true
    @Volatile private var loading = true
    private var lens = CameraSelector.LENS_FACING_BACK
    private var provider: ProcessCameraProvider? = null

    private var photo: Bitmap? = null
    private var photoDets: List<Detection> = emptyList()
    private var pendingSave = false

    // FPS (считается в потоке анализа)
    private var lastFrameT = 0L
    private var fps = 0f
    private var lastUiT = 0L

    private val camPerm = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        if (it) startCamera() else toast("Нужен доступ к камере для режима реального времени")
    }
    private val storagePerm = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        if (it && pendingSave) saveResult() else toast("Нет разрешения на сохранение")
        pendingSave = false
    }
    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) openPhoto(uri)
    }
    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importModel(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityMainBinding.inflate(layoutInflater)
        setContentView(b.root)
        store = ModelStore(this)
        ui = SettingsUi(this)

        syncConfBar()
        b.confSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                val v = max(p, 5)
                b.confText.text = "Порог: $v%"
                if (fromUser) {
                    detector?.conf = v / 100f
                    store.active().s.conf = v / 100f
                }
            }
            override fun onStartTrackingTouch(s: SeekBar?) {}
            override fun onStopTrackingTouch(s: SeekBar?) {
                store.save()
                if (!cameraMode) runPhotoDetection()
            }
        })

        b.btnCamera.setOnClickListener { switchToCamera() }
        b.btnPhoto.setOnClickListener { pickImage.launch("image/*") }
        b.btnFlip.setOnClickListener {
            lens = if (lens == CameraSelector.LENS_FACING_BACK) CameraSelector.LENS_FACING_FRONT
            else CameraSelector.LENS_FACING_BACK
            switchToCamera()
        }
        b.btnClasses.setOnClickListener { showClassDialog() }
        b.btnSave.setOnClickListener { saveResult() }
        b.btnModels.setOnClickListener { showModels() }
        b.btnSettings.setOnClickListener {
            ui.showAppSettings(prefs) { if (cameraMode) startCamera() }
        }

        loadDetector()
        ensureCamera()
    }

    // ---------------- Камера ----------------

    private fun ensureCamera() {
        if (hasCamPerm()) startCamera() else camPerm.launch(Manifest.permission.CAMERA)
    }

    private fun hasCamPerm() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun switchToCamera() {
        cameraMode = true
        b.previewView.visibility = View.VISIBLE
        b.photoView.visibility = View.GONE
        b.btnSave.visibility = View.GONE
        b.overlay.clear()
        ensureCamera()
    }

    /** Выбираем из поддерживаемых камерой диапазонов самый быстрый, не выше запрошенного. */
    @Suppress("UnsafeOptInUsageError")
    private fun pickFpsRange(p: ProcessCameraProvider, sel: CameraSelector, want: Int): Range<Int>? {
        if (want <= 0) return null
        return try {
            val info = sel.filter(p.availableCameraInfos).firstOrNull() ?: return null
            val ranges = Camera2CameraInfo.from(info)
                .getCameraCharacteristic(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
                ?: return null
            ranges.filter { it.upper <= want }
                .maxWithOrNull(compareBy<Range<Int>>({ it.upper }, { it.lower }))
        } catch (_: Throwable) {
            null
        }
    }

    @Suppress("UnsafeOptInUsageError")
    private fun applyFps(pb: Preview.Builder, ab: ImageAnalysis.Builder, r: Range<Int>) {
        Camera2Interop.Extender(pb).setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, r)
        Camera2Interop.Extender(ab).setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, r)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val p = future.get()
            provider = p
            if (!cameraMode) return@addListener
            // модель ещё грузится — камеру запустит loadDetector() по готовности (нужен размер входа для «Авто»)
            if (loading) return@addListener

            var cr = CAM_RES[prefs.getInt("camRes", 0).coerceIn(0, CAM_RES.lastIndex)]
            if (cr.w == 0) {
                val d = detector
                val side = max(d?.inW ?: 0, d?.inH ?: 0)
                cr = if (side in 1..352) CAM_RES[1] else CAM_RES[2]
            }
            val ratio169 = cr.w * 9 == cr.h * 16
            val strat = if (ratio169) AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY
            else AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
            fun sel(w: Int, h: Int) = ResolutionSelector.Builder()
                .setAspectRatioStrategy(strat)
                .setResolutionStrategy(
                    ResolutionStrategy(Size(w, h), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
                ).build()

            val pq = prefs.getInt("prevQ", 0).coerceIn(0, 2)
            val pv = if (ratio169) listOf(Size(640, 360), Size(1280, 720), Size(1920, 1080))[pq]
            else listOf(Size(640, 480), Size(1024, 768), Size(1600, 1200))[pq]

            val pb = Preview.Builder().setResolutionSelector(sel(pv.width, pv.height))
            val ab = ImageAnalysis.Builder()
                .setResolutionSelector(sel(cr.w, cr.h))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            val selector = CameraSelector.Builder().requireLensFacing(lens).build()
            val fr = pickFpsRange(p, selector, prefs.getInt("camFps", 0))
            if (fr != null) {
                try { applyFps(pb, ab, fr) } catch (_: Throwable) {}
            }
            val preview = pb.build().also { it.setSurfaceProvider(b.previewView.surfaceProvider) }
            val analysis = ab.build()

            analysis.setAnalyzer(executor) { proxy ->
                try {
                    val det = detector
                    if (det == null || !cameraMode) return@setAnalyzer
                    val rot = proxy.imageInfo.rotationDegrees
                    val plane = proxy.planes[0]
                    val dets = det.detectRgba(plane.buffer, plane.rowStride, proxy.width, proxy.height, rot)
                    val now = SystemClock.elapsedRealtime()
                    val dt = now - lastFrameT
                    lastFrameT = now
                    if (dt in 1..2000) {
                        val inst = 1000f / dt
                        fps = if (fps == 0f) inst else fps * 0.9f + inst * 0.1f
                    }
                    val swap = rot == 90 || rot == 270
                    val w = if (swap) proxy.height else proxy.width
                    val h = if (swap) proxy.width else proxy.height
                    val mirror = lens == CameraSelector.LENS_FACING_FRONT
                    val updateText = now - lastUiT >= 250
                    if (updateText) lastUiT = now
                    val text = if (updateText) {
                        "${det.backendInfo} · ${det.inW}×${det.inH}\n" +
                            "${f1(fps)} FPS | пре ${f1(det.preMs)} · инф ${f1(det.infMs)} · пост ${f1(det.postMs)} мс | " +
                            "объектов: ${dets.size}"
                    } else null
                    runOnUiThread {
                        if (cameraMode) {
                            b.overlay.setResults(dets, w, h, mirror)
                            if (text != null) b.infoText.text = text
                        }
                    }
                } catch (e: Throwable) {
                    runOnUiThread { b.infoText.text = "Ошибка: ${e.message}" }
                } finally {
                    proxy.close()
                }
            }
            try {
                p.unbindAll()
                p.bindToLifecycle(this, selector, preview, analysis)
            } catch (e: Throwable) {
                toast("Не удалось открыть камеру: ${e.message}")
                if (lens == CameraSelector.LENS_FACING_FRONT) lens = CameraSelector.LENS_FACING_BACK
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun f1(v: Float) = String.format(Locale.US, "%.1f", v)

    // ---------------- Фото ----------------

    private fun openPhoto(uri: Uri) {
        val bmp = loadBitmap(uri)
        if (bmp == null) { toast("Не удалось открыть изображение"); return }
        cameraMode = false
        provider?.unbindAll()
        photo = bmp
        photoDets = emptyList()
        b.photoView.setImageBitmap(bmp)
        b.photoView.visibility = View.VISIBLE
        b.previewView.visibility = View.GONE
        b.btnSave.visibility = View.VISIBLE
        b.overlay.clear()
        runPhotoDetection()
    }

    private fun runPhotoDetection() {
        val bmp = photo ?: return
        val det = detector ?: return
        b.infoText.text = "Распознаю…"
        executor.execute {
            try {
                val t0 = SystemClock.elapsedRealtime()
                val dets = det.detectBitmap(bmp)
                val ms = SystemClock.elapsedRealtime() - t0
                runOnUiThread {
                    if (!cameraMode && photo === bmp) {
                        photoDets = dets
                        b.overlay.setResults(dets, bmp.width, bmp.height, false)
                        b.infoText.text = "${det.backendInfo} · ${det.inW}×${det.inH}\n$ms мс | объектов: ${dets.size}"
                    }
                }
            } catch (e: Throwable) {
                runOnUiThread { b.infoText.text = "Ошибка: ${e.message}" }
            }
        }
    }

    private fun loadBitmap(uri: Uri): Bitmap? {
        return try {
            val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, o) }
            var sample = 1
            while (max(o.outWidth, o.outHeight) / sample > 2048) sample *= 2
            val o2 = BitmapFactory.Options().apply { inSampleSize = sample }
            var bmp = contentResolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it, null, o2)
            } ?: return null
            val orient = contentResolver.openInputStream(uri)?.use {
                ExifInterface(it).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
                )
            } ?: ExifInterface.ORIENTATION_NORMAL
            val m = Matrix()
            when (orient) {
                ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
                ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
                ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
            }
            if (!m.isIdentity) bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            bmp
        } catch (e: Throwable) {
            null
        }
    }

    private fun saveResult() {
        val bmp = photo ?: return
        if (Build.VERSION.SDK_INT < 29 && ContextCompat.checkSelfPermission(
                this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            pendingSave = true
            storagePerm.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            return
        }
        val out = OverlayView.drawOnBitmap(bmp, photoDets)
        try {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "yolo_${System.currentTimeMillis()}.jpg")
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= 29)
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/YoloDetect")
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)!!
            contentResolver.openOutputStream(uri)!!.use { out.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            toast("Сохранено в Pictures/YoloDetect")
        } catch (e: Throwable) {
            toast("Ошибка сохранения: ${e.message}")
        }
    }

    // ---------------- Фильтр классов (хранится в настройках модели) ----------------

    private fun updateClassButton() {
        val names = detector?.names ?: return
        val sel = store.active().s.classes
        b.btnClasses.text = when {
            sel.isEmpty() -> "Классы: все"
            sel.size == 1 -> "Класс: ${names.getOrElse(sel.first()) { "?" }}"
            else -> "Классы: ${sel.size}"
        }
    }

    private fun showClassDialog() {
        val names = detector?.names ?: run { toast("Модель не загружена"); return }
        val temp = store.active().s.classes.toMutableSet()
        var shown: List<Int> = names.indices.toList()
        val dp = resources.displayMetrics.density

        val adapter = object : BaseAdapter() {
            override fun getCount() = shown.size
            override fun getItem(i: Int) = shown[i]
            override fun getItemId(i: Int) = shown[i].toLong()
            override fun getView(i: Int, convert: View?, parent: ViewGroup): View {
                val cb = (convert as? CheckBox) ?: CheckBox(this@MainActivity).apply {
                    textSize = 16f
                    setPadding((8 * dp).toInt(), (10 * dp).toInt(), (8 * dp).toInt(), (10 * dp).toInt())
                }
                val id = shown[i]
                cb.text = names[id]
                cb.isChecked = id in temp
                cb.setOnClickListener {
                    if (cb.isChecked) temp.add(id) else temp.remove(id)
                }
                return cb
            }
        }

        val search = EditText(this).apply { hint = "Поиск класса (например: person)"; isSingleLine = true }
        val list = ListView(this).apply { this.adapter = adapter }
        val btnAll = Button(this).apply { text = "Все классы" }
        val btnClear = Button(this).apply { text = "Сбросить" }
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(btnAll, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(btnClear, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding((16 * dp).toInt(), (8 * dp).toInt(), (16 * dp).toInt(), 0)
            addView(search)
            addView(row)
            addView(list, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, (360 * dp).toInt()))
        }

        search.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) {
                val q = s?.toString()?.trim()?.lowercase().orEmpty()
                shown = names.indices.filter { q.isEmpty() || names[it].lowercase().contains(q) }
                adapter.notifyDataSetChanged()
            }
            override fun beforeTextChanged(s: CharSequence?, st: Int, c: Int, a: Int) {}
            override fun onTextChanged(s: CharSequence?, st: Int, bf: Int, c: Int) {}
        })
        btnAll.setOnClickListener { temp.clear(); adapter.notifyDataSetChanged() }
        btnClear.setOnClickListener { temp.clear(); adapter.notifyDataSetChanged() }

        AlertDialog.Builder(this)
            .setTitle("Показывать только:")
            .setMessage("Ничего не отмечено = все классы")
            .setView(root)
            .setPositiveButton("Готово") { _, _ ->
                val set = temp.toSet()
                store.active().s.classes = set
                detector?.allowed = set
                store.save()
                updateClassButton()
                if (!cameraMode) runPhotoDetection()
            }
            .setNegativeButton("Отмена", null)
            .show()
    }

    // ---------------- Модели ----------------

    private fun syncConfBar() {
        val v = (store.active().s.conf * 100).roundToInt().coerceIn(5, 95)
        b.confSeek.progress = v
        b.confText.text = "Порог: $v%"
    }

    /** Загружает активную модель в фоне; старая работает, пока новая не готова. */
    private fun loadDetector() {
        val m = store.active()
        loading = true
        b.infoText.text = "Загрузка модели «${m.name}»…"
        executor.execute {
            try {
                val old = detector
                val d = Detector(this, m)
                d.warmUp()
                detector = d
                old?.close()
                loading = false
                runOnUiThread {
                    b.infoText.text = "${m.name}: ${d.backendInfo}, классов: ${d.names.size}"
                    syncConfBar()
                    updateClassButton()
                    if (cameraMode && hasCamPerm()) startCamera()
                    if (!cameraMode) runPhotoDetection()
                }
            } catch (e: Throwable) {
                loading = false
                runOnUiThread {
                    b.infoText.text = "Ошибка загрузки «${m.name}»: ${e.message}"
                    if (!m.builtin) {
                        toast("Не удалось загрузить «${m.name}», включаю встроенную модель")
                        store.setActive(ModelStore.BUILTIN_ID)
                        loadDetector()
                    } else if (cameraMode && hasCamPerm()) startCamera()
                }
            }
        }
    }

    private fun selectModel(m: ModelEntry) {
        if (m.id == store.activeId) return
        store.setActive(m.id)
        loadDetector()
    }

    private fun showModels() {
        ui.showModels(
            store,
            onSelect = { selectModel(it) },
            onConfigure = { m ->
                ui.showModelSettings(m, m.id == store.activeId) { changed ->
                    store.save()
                    if (changed.id == store.activeId) loadDetector()
                    ui.refreshModels()
                }
            },
            onDelete = { m ->
                AlertDialog.Builder(this)
                    .setTitle("Удалить «${m.name}»?")
                    .setMessage("Файл модели и её настройки будут удалены.")
                    .setPositiveButton("Удалить") { _, _ ->
                        val wasActive = m.id == store.activeId
                        store.remove(m)
                        ui.refreshModels()
                        if (wasActive) loadDetector()
                    }
                    .setNegativeButton("Отмена", null)
                    .show()
            },
            onAdd = { pickModel.launch(arrayOf("*/*")) }
        )
    }

    private fun displayName(uri: Uri): String {
        try {
            contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) return c.getString(0) ?: ""
            }
        } catch (_: Throwable) {
        }
        return uri.lastPathSegment ?: ""
    }

    /** Копирует выбранный .onnx в память приложения, проверяет его и добавляет в список. */
    private fun importModel(uri: Uri) {
        val fileName = displayName(uri)
        val lower = fileName.lowercase()
        if (lower.endsWith(".pt") || lower.endsWith(".pth")) {
            toast("Нужен формат .onnx. Файл .pt сначала конвертируй (workflow «Convert model» в GitHub или команда yolo export, см. README)")
            return
        }
        b.infoText.text = "Импорт модели…"
        val id = ModelStore.newId()
        val entry = ModelEntry(
            id = id,
            name = fileName.substringBeforeLast('.').ifEmpty { "Модель" },
            builtin = false,
            file = "$id.onnx",
            s = ModelSettings(backend = Backend.CPU, imgsz = 640)
        )
        executor.execute {
            val f: File = ModelStore.modelFile(this, entry)
            try {
                contentResolver.openInputStream(uri)!!.use { input ->
                    f.outputStream().use { out -> input.copyTo(out, 256 * 1024) }
                }
                entry.sizeBytes = f.length()
                // пробная загрузка: проверяем, что это рабочая модель, и узнаём вход/число классов
                val d = Detector(this, entry)
                entry.fixedW = d.fixedW
                entry.fixedH = d.fixedH
                entry.nc = if (d.ncHint > 0) d.ncHint else d.names.size
                d.close()
                runOnUiThread {
                    store.add(entry)
                    store.setActive(entry.id)
                    toast("Модель «${entry.name}» добавлена")
                    loadDetector()
                }
            } catch (e: Throwable) {
                try { f.delete() } catch (_: Throwable) {}
                runOnUiThread {
                    b.infoText.text = "Не удалось добавить модель"
                    toast("Не удалось добавить модель: ${e.message}")
                }
            }
        }
    }

    private fun toast(s: String) = Toast.makeText(this, s, Toast.LENGTH_LONG).show()

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        detector?.close()
    }
}
