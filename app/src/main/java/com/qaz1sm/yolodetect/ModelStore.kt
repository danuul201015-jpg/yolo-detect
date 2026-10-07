package com.qaz1sm.yolodetect

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

enum class EngineType { FAST, CLASSIC }

enum class Backend { CPU, XNNPACK, NNAPI }

/** Настройки одной модели. Хранятся вместе с моделью. */
data class ModelSettings(
    var engine: EngineType = EngineType.FAST,
    var backend: Backend = Backend.XNNPACK,
    var fp16: Boolean = true,          // NNAPI: режим FP16
    var threads: Int = 4,
    var imgsz: Int = 320,              // разрешение входа (для встроенной — из списка, для динамических ONNX — кратно 32)
    var rect: Boolean = true,          // прямоугольный вход под пропорции кадра (только динамические модели)
    var conf: Float = 0.40f,
    var iou: Float = 0.45f,
    var maxDet: Int = 100,
    var classes: Set<Int> = emptySet(), // пусто = все классы
    var names: List<String> = emptyList() // свои названия классов (пусто = из модели)
)

class ModelEntry(
    val id: String,
    var name: String,
    val builtin: Boolean,
    val file: String,                  // имя файла в filesDir/models (для встроенной пусто)
    var fixedW: Int = 0,               // 0 = динамический вход
    var fixedH: Int = 0,
    var nc: Int = 0,
    var sizeBytes: Long = 0,
    var s: ModelSettings = ModelSettings()
)

class ModelStore(private val ctx: Context) {
    private val prefs = ctx.getSharedPreferences("models", Context.MODE_PRIVATE)
    val models = ArrayList<ModelEntry>()
    var activeId: String = BUILTIN_ID
        private set

    init { load() }

    fun active(): ModelEntry = models.firstOrNull { it.id == activeId } ?: models.first()

    fun setActive(id: String) {
        if (models.any { it.id == id }) activeId = id else activeId = BUILTIN_ID
        prefs.edit().putString("active", activeId).apply()
    }

    fun add(e: ModelEntry) {
        models.add(e)
        save()
    }

    /** Удаляет пользовательскую модель вместе с файлом. Встроенную удалить нельзя. */
    fun remove(e: ModelEntry) {
        if (e.builtin) return
        models.remove(e)
        try { modelFile(ctx, e).delete() } catch (_: Throwable) {}
        if (activeId == e.id) activeId = BUILTIN_ID
        save()
        prefs.edit().putString("active", activeId).apply()
    }

    fun save() {
        val arr = JSONArray()
        for (m in models) arr.put(toJson(m))
        prefs.edit().putString("json", arr.toString()).apply()
    }

    private fun load() {
        models.clear()
        val raw = prefs.getString("json", null)
        if (raw != null) {
            try {
                val arr = JSONArray(raw)
                for (i in 0 until arr.length()) {
                    val e = fromJson(arr.getJSONObject(i))
                    if (e.builtin || modelFile(ctx, e).exists()) models.add(e)
                }
            } catch (_: Throwable) {
            }
        }
        if (models.none { it.builtin }) models.add(0, builtinEntry())
        activeId = prefs.getString("active", BUILTIN_ID) ?: BUILTIN_ID
        if (models.none { it.id == activeId }) activeId = BUILTIN_ID
    }

    private fun toJson(m: ModelEntry): JSONObject {
        val s = m.s
        return JSONObject().apply {
            put("id", m.id); put("name", m.name); put("builtin", m.builtin); put("file", m.file)
            put("fixedW", m.fixedW); put("fixedH", m.fixedH); put("nc", m.nc); put("size", m.sizeBytes)
            put("engine", s.engine.name); put("backend", s.backend.name); put("fp16", s.fp16)
            put("threads", s.threads); put("imgsz", s.imgsz); put("rect", s.rect)
            put("conf", s.conf.toDouble()); put("iou", s.iou.toDouble()); put("maxDet", s.maxDet)
            put("classes", JSONArray(s.classes.toList()))
            put("names", JSONArray(s.names))
        }
    }

    private fun fromJson(o: JSONObject): ModelEntry {
        val s = ModelSettings(
            engine = runCatching { EngineType.valueOf(o.optString("engine")) }.getOrDefault(EngineType.FAST),
            backend = runCatching { Backend.valueOf(o.optString("backend")) }.getOrDefault(Backend.CPU),
            fp16 = o.optBoolean("fp16", true),
            threads = o.optInt("threads", 4).coerceIn(1, 8),
            imgsz = o.optInt("imgsz", 320),
            rect = o.optBoolean("rect", true),
            conf = o.optDouble("conf", 0.40).toFloat().coerceIn(0.05f, 0.95f),
            iou = o.optDouble("iou", 0.45).toFloat().coerceIn(0.05f, 0.95f),
            maxDet = o.optInt("maxDet", 100).coerceIn(1, 1000),
            classes = o.optJSONArray("classes")?.let { a -> (0 until a.length()).map { a.getInt(it) }.toSet() }
                ?: emptySet(),
            names = o.optJSONArray("names")?.let { a -> (0 until a.length()).map { a.getString(it) } }
                ?: emptyList()
        )
        return ModelEntry(
            id = o.getString("id"), name = o.optString("name", "Модель"),
            builtin = o.optBoolean("builtin", false), file = o.optString("file", ""),
            fixedW = o.optInt("fixedW", 0), fixedH = o.optInt("fixedH", 0), nc = o.optInt("nc", 0),
            sizeBytes = o.optLong("size", 0), s = s
        )
    }

    companion object {
        const val BUILTIN_ID = "builtin"
        /** Размеры, под которые встроенная модель экспортируется в GitHub Actions. */
        val BUILTIN_SIZES = listOf(256, 320, 416, 480, 640)

        fun builtinEntry() = ModelEntry(BUILTIN_ID, "YOLO26n (встроенная)", true, "")

        fun dir(ctx: Context): File = File(ctx.filesDir, "models").also { it.mkdirs() }

        fun modelFile(ctx: Context, e: ModelEntry): File = File(dir(ctx), e.file)

        fun newId(): String = "m" + System.currentTimeMillis().toString(36)

        fun snapBuiltin(size: Int): Int = BUILTIN_SIZES.minByOrNull { kotlin.math.abs(it - size) } ?: 320
    }
}
