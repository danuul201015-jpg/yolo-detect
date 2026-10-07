package com.qaz1sm.yolodetect

import android.content.SharedPreferences
import android.graphics.Typeface
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity

/** Пресеты разрешения анализа камеры (в ландшафтной ориентации сенсора). 0×0 = авто. */
class CamRes(val label: String, val w: Int, val h: Int)

val CAM_RES = listOf(
    CamRes("Авто (под размер входа модели)", 0, 0),
    CamRes("320×240 (4:3) — максимум FPS", 320, 240),
    CamRes("640×480 (4:3)", 640, 480),
    CamRes("800×600 (4:3)", 800, 600),
    CamRes("1280×960 (4:3)", 1280, 960),
    CamRes("640×360 (16:9)", 640, 360),
    CamRes("1280×720 (16:9)", 1280, 720),
    CamRes("1920×1080 (16:9)", 1920, 1080)
)
val PREVIEW_Q = listOf("Низкое (быстрее)", "Среднее", "Высокое")
val CAM_FPS = listOf(0 to "Авто", 30 to "До 30", 60 to "До 60", 120 to "До 120")

private class LabeledSeek(
    ui: SettingsUi, private val lo: Int, hi: Int, v: Int, private val fmt: (Int) -> String
) {
    val label: TextView = ui.label(fmt(v))
    val bar: SeekBar = SeekBar(ui.a)
    val value: Int get() = bar.progress + lo

    init {
        bar.max = hi - lo
        bar.progress = v - lo
        bar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, u: Boolean) { label.text = fmt(p + lo) }
            override fun onStartTrackingTouch(s: SeekBar?) {}
            override fun onStopTrackingTouch(s: SeekBar?) {}
        })
    }

    fun addTo(col: LinearLayout) { col.addView(label); col.addView(bar) }
}

class SettingsUi(val a: AppCompatActivity) {
    private val dp = a.resources.displayMetrics.density
    private fun px(v: Int) = (v * dp).toInt()

    private var modelsDialog: AlertDialog? = null
    private var modelsAdapter: BaseAdapter? = null

    fun label(t: String) = TextView(a).apply {
        text = t; textSize = 15f; setPadding(0, px(14), 0, px(4))
    }

    private fun hint(t: String) = TextView(a).apply {
        text = t; textSize = 12f; alpha = 0.7f; setPadding(0, px(4), 0, 0)
    }

    private fun column() = LinearLayout(a).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(px(20), px(4), px(20), px(8))
    }

    private fun radios(items: List<String>, checked: Int): RadioGroup {
        val rg = RadioGroup(a)
        items.forEachIndexed { i, t ->
            rg.addView(RadioButton(a).apply { text = t; id = 1000 + i; isChecked = i == checked })
        }
        return rg
    }

    private fun RadioGroup.index(max: Int) = (checkedRadioButtonId - 1000).coerceIn(0, max)

    // ------------------------------------------------------------------ список моделей

    fun describe(m: ModelEntry): String {
        val s = m.s
        val fast = s.engine == EngineType.FAST
        val eng = if (fast) "Быстрый" else "Классический"
        val be = when (s.backend) {
            Backend.CPU -> "CPU"
            Backend.XNNPACK -> if (fast) "XNNPACK" else "CPU"
            Backend.NNAPI -> "NNAPI"
        }
        val res = when {
            m.builtin -> "${ModelStore.snapBuiltin(s.imgsz)}px"
            m.fixedW > 0 && m.fixedH > 0 -> "${m.fixedW}×${m.fixedH}"
            else -> "${s.imgsz}px" + if (s.rect) " (прям.)" else ""
        }
        val cls = if (m.nc > 0) " · ${m.nc} кл." else ""
        val size = if (!m.builtin && m.sizeBytes > 0) " · ${m.sizeBytes / 1048576} МБ" else ""
        return "$eng · $be · $res · ${s.threads} пот.$cls$size"
    }

    fun showModels(
        store: ModelStore,
        onSelect: (ModelEntry) -> Unit,
        onConfigure: (ModelEntry) -> Unit,
        onDelete: (ModelEntry) -> Unit,
        onAdd: () -> Unit
    ) {
        val adapter = object : BaseAdapter() {
            override fun getCount() = store.models.size
            override fun getItem(i: Int) = store.models[i]
            override fun getItemId(i: Int) = i.toLong()
            override fun getView(i: Int, convert: View?, parent: ViewGroup): View {
                val m = store.models[i]
                val active = m.id == store.activeId
                val row = LinearLayout(a).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(px(12), px(8), px(4), px(8))
                }
                val col = LinearLayout(a).apply { orientation = LinearLayout.VERTICAL }
                col.addView(TextView(a).apply {
                    text = (if (active) "● " else "○ ") + m.name
                    textSize = 16f
                    if (active) setTypeface(typeface, Typeface.BOLD)
                })
                col.addView(TextView(a).apply { text = describe(m); textSize = 12f; alpha = 0.7f })
                row.addView(col, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                row.addView(TextView(a).apply {
                    text = "⚙"; textSize = 22f; setPadding(px(14), px(8), px(14), px(8))
                    setOnClickListener { onConfigure(m) }
                })
                if (!m.builtin) {
                    row.addView(TextView(a).apply {
                        text = "✕"; textSize = 20f; setPadding(px(14), px(8), px(14), px(8))
                        setOnClickListener { onDelete(m) }
                    })
                }
                row.setOnClickListener { modelsDialog?.dismiss(); onSelect(m) }
                return row
            }
        }
        modelsAdapter = adapter
        val list = ListView(a).apply { this.adapter = adapter }
        val btnAdd = Button(a).apply { text = "＋ Добавить модель (.onnx)"; setOnClickListener { modelsDialog?.dismiss(); onAdd() } }
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(px(12), px(4), px(12), 0)
            addView(list, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, px(320)))
            addView(btnAdd)
            addView(hint("Нажми на модель — сделать активной. ⚙ — настройки этой модели. " +
                "Свои модели хранятся в приложении и остаются в списке."))
        }
        modelsDialog = AlertDialog.Builder(a)
            .setTitle("Модели")
            .setView(root)
            .setNegativeButton("Закрыть", null)
            .show()
    }

    fun refreshModels() { modelsAdapter?.notifyDataSetChanged() }

    // ------------------------------------------------------------------ настройки модели

    fun showModelSettings(m: ModelEntry, isActive: Boolean, onApply: (ModelEntry) -> Unit) {
        val s = m.s
        val col = column()

        val etName = EditText(a).apply { setText(m.name); isSingleLine = true; hint = "Название" }
        col.addView(label("Название")); col.addView(etName)

        col.addView(label("Движок"))
        val rgEngine = radios(
            listOf("Быстрый (новый, оптимизированный)", "Классический (старый)"),
            if (s.engine == EngineType.FAST) 0 else 1
        )
        col.addView(rgEngine)
        col.addView(hint("Быстрый: подготовка кадра по готовым таблицам, чтение выхода без копирования, " +
            "расширенная оптимизация графа, XNNPACK. Классический — прежний режим, если на устройстве что-то нестабильно."))

        col.addView(label("Где считать"))
        val rgBackend = radios(
            listOf("CPU (процессор)", "XNNPACK — ускоренный CPU ARM (только быстрый движок)", "NNAPI — GPU/NPU/APU"),
            s.backend.ordinal
        )
        col.addView(rgBackend)
        val cbFp16 = CheckBox(a).apply { text = "NNAPI: режим FP16 (быстрее)"; isChecked = s.fp16 }
        col.addView(cbFp16)
        col.addView(hint("Если выбранный вариант недоступен, приложение само переключится на CPU — это видно в верхней плашке."))

        val sThreads = LabeledSeek(this, 1, 8, s.threads.coerceIn(1, 8)) { "Потоки CPU: $it" }
        sThreads.addTo(col)

        // --- разрешение входа
        var rgSize: RadioGroup? = null
        var sSize: LabeledSeek? = null
        var cbRect: CheckBox? = null
        if (m.builtin) {
            col.addView(label("Разрешение входа"))
            val sizes = ModelStore.BUILTIN_SIZES
            rgSize = radios(sizes.map { "$it px" }, sizes.indexOf(ModelStore.snapBuiltin(s.imgsz)).coerceAtLeast(0))
            col.addView(rgSize)
            col.addView(hint("Меньше — быстрее, больше — точнее на мелких объектах."))
        } else if (m.fixedW > 0 && m.fixedH > 0) {
            col.addView(label("Разрешение входа"))
            col.addView(hint("У этой модели вход фиксирован: ${m.fixedW}×${m.fixedH}. " +
                "Чтобы менять разрешение, экспортируй её в ONNX с dynamic=True."))
        } else {
            sSize = LabeledSeek(this, 2, 40, (s.imgsz / 32).coerceIn(2, 40)) { "Разрешение входа: ${it * 32} px" }
            sSize.addTo(col)
            cbRect = CheckBox(a).apply { text = "Прямоугольный вход под пропорции кадра (быстрее)"; isChecked = s.rect }
            col.addView(cbRect)
        }

        val sIou = LabeledSeek(this, 10, 90, (s.iou * 100).toInt().coerceIn(10, 90)) { "Порог NMS (IoU): $it%" }
        sIou.addTo(col)
        val sMax = LabeledSeek(this, 1, 300, s.maxDet.coerceIn(1, 300)) { "Макс. объектов на кадр: $it" }
        sMax.addTo(col)
        val sConf = LabeledSeek(this, 5, 95, (s.conf * 100).toInt().coerceIn(5, 95)) { "Порог уверенности: $it%" }
        sConf.addTo(col)

        col.addView(label("Названия классов"))
        val etNames = EditText(a).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            minLines = 2
            hint = "по одному в строке или через запятую; пусто — из модели"
            setText(s.names.joinToString("\n"))
        }
        col.addView(etNames)

        val finalRgSize = rgSize; val finalSSize = sSize; val finalCbRect = cbRect
        AlertDialog.Builder(a)
            .setTitle("Модель: ${m.name}")
            .setView(ScrollView(a).apply { addView(col) })
            .setPositiveButton("Применить") { _, _ ->
                s.engine = if (rgEngine.index(1) == 0) EngineType.FAST else EngineType.CLASSIC
                s.backend = Backend.values()[rgBackend.index(2)]
                s.fp16 = cbFp16.isChecked
                s.threads = sThreads.value
                if (finalRgSize != null) s.imgsz = ModelStore.BUILTIN_SIZES[finalRgSize.index(ModelStore.BUILTIN_SIZES.lastIndex)]
                if (finalSSize != null) s.imgsz = finalSSize.value * 32
                if (finalCbRect != null) s.rect = finalCbRect.isChecked
                s.iou = sIou.value / 100f
                s.maxDet = sMax.value
                s.conf = sConf.value / 100f
                s.names = etNames.text.toString().split('\n', ',').map { it.trim() }.filter { it.isNotEmpty() }
                m.name = etName.text.toString().trim().ifEmpty { m.name }
                onApply(m)
            }
            .setNeutralButton("Сброс") { _, _ ->
                val keepClasses = s.classes
                m.s = ModelSettings(imgsz = if (m.builtin) 320 else 640).also { it.classes = keepClasses }
                onApply(m)
            }
            .setNegativeButton("Отмена", null)
            .show()
    }

    // ------------------------------------------------------------------ настройки приложения (камера)

    fun showAppSettings(prefs: SharedPreferences, onApply: () -> Unit) {
        val col = column()
        col.addView(label("Разрешение анализа камеры"))
        val rgRes = radios(CAM_RES.map { it.label }, prefs.getInt("camRes", 0).coerceIn(0, CAM_RES.lastIndex))
        col.addView(rgRes)
        col.addView(hint("Меньше — выше FPS. Камера отдаёт кадр ближайшего поддерживаемого размера."))

        col.addView(label("Качество превью"))
        val rgPrev = radios(PREVIEW_Q, prefs.getInt("prevQ", 0).coerceIn(0, 2))
        col.addView(rgPrev)
        col.addView(hint("Влияет только на картинку на экране, не на распознавание."))

        col.addView(label("Частота кадров камеры"))
        val cur = CAM_FPS.indexOfFirst { it.first == prefs.getInt("camFps", 0) }.coerceAtLeast(0)
        val rgFps = radios(CAM_FPS.map { it.second }, cur)
        col.addView(rgFps)
        col.addView(hint("Выше 30 работает, только если камера телефона это поддерживает. Настройки конкретной модели — в списке «Модели» (⚙)."))

        AlertDialog.Builder(a)
            .setTitle("Настройки")
            .setView(ScrollView(a).apply { addView(col) })
            .setPositiveButton("Применить") { _, _ ->
                prefs.edit()
                    .putInt("camRes", rgRes.index(CAM_RES.lastIndex))
                    .putInt("prevQ", rgPrev.index(2))
                    .putInt("camFps", CAM_FPS[rgFps.index(CAM_FPS.lastIndex)].first)
                    .apply()
                onApply()
            }
            .setNegativeButton("Отмена", null)
            .show()
    }
}
