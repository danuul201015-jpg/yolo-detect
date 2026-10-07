package com.qaz1sm.yolodetect

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Arrays
import java.util.EnumSet
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/** Рамка в нормализованных координатах (0..1) относительно (повёрнутого) изображения. */
data class Detection(
    val left: Float, val top: Float, val right: Float, val bottom: Float,
    val classId: Int, val label: String, val score: Float
)

/**
 * Детектор на ONNX Runtime. Два движка:
 *  - CLASSIC — прежний путь (поэлементная запись в буфер, вывод через вложенные массивы);
 *  - FAST — таблицы пересчёта координат, LUT, массовое копирование, чтение вывода без вложенных массивов,
 *    расширенная оптимизация графа, опционально XNNPACK.
 * Вход может быть динамическим (любое разрешение, кратное 32) или фиксированным (берётся из модели).
 */
class Detector(context: Context, entry: ModelEntry) {
    private val env = OrtEnvironment.getEnvironment()
    private val st: ModelSettings = entry.s
    private val fast = st.engine == EngineType.FAST
    private val builtin = entry.builtin
    private val session: OrtSession
    private val inputName: String

    /** 0 = ось динамическая. */
    val fixedW: Int
    val fixedH: Int

    /** Число классов по форме выхода (0 — неизвестно, напр. end-to-end модель без метаданных). */
    val ncHint: Int
    val names: List<String>
    private val metaCount: Int

    /** Что реально используется (при ошибке XNNPACK/NNAPI автоматически падаем на CPU). */
    val backendInfo: String

    @Volatile var conf: Float = st.conf
    @Volatile var iou: Float = st.iou
    @Volatile var maxDet: Int = st.maxDet
    @Volatile var allowed: Set<Int> = st.classes

    var inW = 0
        private set
    var inH = 0
        private set

    // тайминги последнего кадра, мс
    @Volatile var preMs = 0f
        private set
    @Volatile var infMs = 0f
        private set
    @Volatile var postMs = 0f
        private set

    private var inputBuf: FloatBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private var fArr = FloatArray(0)
    private var srcBytes = ByteArray(0)
    private var srcOff = IntArray(0)
    private var tabKey = ""
    private var outArr = FloatArray(0)
    private var bestS = FloatArray(0)
    private var bestC = IntArray(0)
    private val lut = FloatArray(256) { it / 255f }

    init {
        val bytes: ByteArray? = if (builtin) {
            val sz = ModelStore.snapBuiltin(st.imgsz)
            context.assets.open("model_$sz.onnx").use { it.readBytes() }
        } else null
        val path: String? = if (builtin) null else ModelStore.modelFile(context, entry).absolutePath

        fun create(o: OrtSession.SessionOptions): OrtSession =
            if (bytes != null) env.createSession(bytes, o) else env.createSession(path!!, o)

        var tag = "CPU"
        var made: OrtSession? = null
        if (st.backend == Backend.NNAPI) {
            try {
                val o = options(st.threads)
                val flags = if (st.fp16) EnumSet.of(NNAPIFlags.USE_FP16) else EnumSet.noneOf(NNAPIFlags::class.java)
                o.addNnapi(flags)
                made = create(o)
                tag = "NNAPI"
            } catch (e: Throwable) {
                made = null
                tag = "CPU (NNAPI недоступен)"
            }
        } else if (st.backend == Backend.XNNPACK && fast) {
            try {
                val o = options(1)
                // через рефлексию: если в сборке ORT нет метода/провайдера — просто падаем на CPU
                o.javaClass.getMethod("addXnnpack", Map::class.java)
                    .invoke(o, mapOf("intra_op_num_threads" to st.threads.toString()))
                made = create(o)
                tag = "XNNPACK"
            } catch (e: Throwable) {
                made = null
                tag = "CPU (XNNPACK недоступен)"
            }
        }
        val s: OrtSession = made ?: create(options(st.threads))
        session = s
        backendInfo = (if (fast) "Быстрый" else "Классич.") + " · " + tag

        inputName = s.inputNames.first()
        val ishape = (s.inputInfo[inputName]!!.info as TensorInfo).shape
        fixedH = if (ishape.size >= 4 && ishape[2] > 0) ishape[2].toInt() else 0
        fixedW = if (ishape.size >= 4 && ishape[3] > 0) ishape[3].toInt() else 0

        val oshape = (s.outputInfo.values.first().info as TensorInfo).shape
        ncHint = outNc(oshape)

        val meta = readMeta(s)
        metaCount = meta.size
        names = when {
            st.names.isNotEmpty() -> st.names
            meta.isNotEmpty() -> meta
            ncHint == 80 || (ncHint <= 0 && builtin) -> COCO
            else -> List(if (ncHint > 0) ncHint else 80) { "class $it" }
        }
    }

    private fun options(intra: Int): OrtSession.SessionOptions {
        val o = OrtSession.SessionOptions()
        o.setIntraOpNumThreads(intra)
        o.setInterOpNumThreads(1)
        if (fast) {
            try {
                o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                o.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                o.setMemoryPatternOptimization(true)
                o.addConfigEntry("session.intra_op.allow_spinning", "1")
            } catch (_: Throwable) {
            }
        }
        return o
    }

    private fun outNc(s: LongArray): Int {
        if (s.size != 3) return 0
        if (s[2] == 6L && s[1] != 6L) return 0 // end-to-end [N, 6]
        val ch = if (s[2] <= 0L) s[1] else if (s[1] <= 0L) s[2] else min(s[1], s[2])
        return if (ch > 4) (ch - 4).toInt() else 0
    }

    private fun readMeta(s: OrtSession): List<String> {
        try {
            val raw = s.metadata.customMetadata["names"]
            if (raw != null) {
                val map = sortedMapOf<Int, String>()
                Regex("(\\d+):\\s*(?:'([^']*)'|\"([^\"]*)\")").findAll(raw).forEach { m ->
                    map[m.groupValues[1].toInt()] = m.groupValues[2].ifEmpty { m.groupValues[3] }
                }
                if (map.isNotEmpty()) return (0..map.lastKey()).map { map[it] ?: "class$it" }
            }
        } catch (_: Throwable) {
        }
        return emptyList()
    }

    @Synchronized
    fun close() {
        try { session.close() } catch (_: Throwable) {}
    }

    // ---------------------------------------------------------------- размеры входа

    /** [w, h] входа для кадра rw×rh. */
    private fun targetSize(rw: Int, rh: Int): IntArray {
        var w: Int
        var h: Int
        val base = max(32, (st.imgsz.coerceIn(64, 1920) / 32) * 32)
        if (!st.rect) {
            w = base; h = base
        } else {
            val lng = max(rw, rh).toFloat()
            val sht = min(rw, rh).toFloat()
            val s = max(32, ceil(base * sht / lng / 32f).toInt() * 32)
            if (rw >= rh) { w = base; h = s } else { w = s; h = base }
        }
        if (fixedW > 0) w = fixedW
        if (fixedH > 0) h = fixedH
        return intArrayOf(w, h)
    }

    private fun ensureBuf(w: Int, h: Int) {
        if (w == inW && h == inH) return
        inW = w; inH = h
        inputBuf = ByteBuffer.allocateDirect(4 * 3 * w * h).order(ByteOrder.nativeOrder()).asFloatBuffer()
        fArr = FloatArray(3 * w * h)
        Arrays.fill(fArr, GRAY)
        tabKey = ""
    }

    /** Прогрев: первый запуск ORT всегда медленнее. */
    @Synchronized
    fun warmUp() {
        try {
            val ts = targetSize(640, 480)
            ensureBuf(ts[0], ts[1])
            inputBuf.rewind()
            val t = OnnxTensor.createTensor(env, inputBuf, longArrayOf(1, 3, inH.toLong(), inW.toLong()))
            t.use { session.run(mapOf(inputName to t)).close() }
        } catch (_: Throwable) {
        }
    }

    // ---------------------------------------------------------------- камера

    /**
     * Быстрый путь для камеры: берём RGBA-буфер кадра напрямую, поворот + letterbox + нормализация
     * за один проход, без создания Bitmap. Возвращает рамки, нормализованные к ПОВЁРНУТОМУ кадру.
     */
    @Synchronized
    fun detectRgba(buf: ByteBuffer, rowStride: Int, sw: Int, sh: Int, rot: Int): List<Detection> {
        val t0 = System.nanoTime()
        val rw = if (rot == 90 || rot == 270) sh else sw
        val rh = if (rot == 90 || rot == 270) sw else sh
        val ts = targetSize(rw, rh)
        ensureBuf(ts[0], ts[1])
        val scale = min(inW / rw.toFloat(), inH / rh.toFloat())
        val nw = (rw * scale).toInt().coerceAtLeast(1)
        val nh = (rh * scale).toInt().coerceAtLeast(1)
        val padX = (inW - nw) / 2
        val padY = (inH - nh) / 2
        if (fast) fillRgbaFast(buf, rowStride, sw, sh, rot, rw, rh, scale, nw, nh, padX, padY)
        else fillRgbaClassic(buf, rowStride, sw, sh, rot, rw, rh, scale, nw, nh, padX, padY)
        preMs = (System.nanoTime() - t0) / 1e6f
        return infer(rw, rh, scale, padX, padY)
    }

    private fun fillRgbaFast(
        buf: ByteBuffer, rowStride: Int, sw: Int, sh: Int, rot: Int,
        rw: Int, rh: Int, scale: Float, nw: Int, nh: Int, padX: Int, padY: Int
    ) {
        val plane = inW * inH
        val key = "$sw,$sh,$rowStride,$rot,$inW,$inH"
        if (key != tabKey) {
            tabKey = key
            if (srcOff.size != plane) srcOff = IntArray(plane)
            Arrays.fill(fArr, GRAY)
            val rxOf = IntArray(nw) { (it / scale).toInt().coerceIn(0, rw - 1) }
            val ryOf = IntArray(nh) { (it / scale).toInt().coerceIn(0, rh - 1) }
            for (y in 0 until nh) {
                val ry = ryOf[y]
                val base = (padY + y) * inW + padX
                for (x in 0 until nw) {
                    val rx = rxOf[x]
                    val sx: Int
                    val sy: Int
                    when (rot) {
                        90 -> { sx = ry; sy = sh - 1 - rx }
                        180 -> { sx = sw - 1 - rx; sy = sh - 1 - ry }
                        270 -> { sx = sw - 1 - ry; sy = rx }
                        else -> { sx = rx; sy = ry }
                    }
                    srcOff[base + x] = sy * rowStride + sx * 4
                }
            }
        }
        val d = buf.duplicate()
        d.clear()
        val need = d.remaining()
        if (srcBytes.size < need) srcBytes = ByteArray(need)
        d.get(srcBytes, 0, need)

        val f = fArr
        val l = lut
        val so = srcOff
        val bytes = srcBytes
        val p2 = plane * 2
        for (y in 0 until nh) {
            val base = (padY + y) * inW + padX
            for (x in base until base + nw) {
                val o = so[x]
                f[x] = l[bytes[o].toInt() and 0xFF]
                f[plane + x] = l[bytes[o + 1].toInt() and 0xFF]
                f[p2 + x] = l[bytes[o + 2].toInt() and 0xFF]
            }
        }
        inputBuf.clear()
        inputBuf.put(f)
        inputBuf.rewind()
    }

    private fun fillRgbaClassic(
        buf: ByteBuffer, rowStride: Int, sw: Int, sh: Int, rot: Int,
        rw: Int, rh: Int, scale: Float, nw: Int, nh: Int, padX: Int, padY: Int
    ) {
        val rxOf = IntArray(inW) {
            if (it < padX || it >= padX + nw) -1 else ((it - padX) / scale).toInt().coerceIn(0, rw - 1)
        }
        val ryOf = IntArray(inH) {
            if (it < padY || it >= padY + nh) -1 else ((it - padY) / scale).toInt().coerceIn(0, rh - 1)
        }
        val plane = inW * inH
        val gray = GRAY
        val k = 1f / 255f
        val fb = inputBuf
        for (dy in 0 until inH) {
            val ry = ryOf[dy]
            val rowBase = dy * inW
            for (dx in 0 until inW) {
                val rx = rxOf[dx]
                val i = rowBase + dx
                if (rx < 0 || ry < 0) {
                    fb.put(i, gray); fb.put(plane + i, gray); fb.put(2 * plane + i, gray)
                } else {
                    val sx: Int
                    val sy: Int
                    when (rot) {
                        90 -> { sx = ry; sy = sh - 1 - rx }
                        180 -> { sx = sw - 1 - rx; sy = sh - 1 - ry }
                        270 -> { sx = sw - 1 - ry; sy = rx }
                        else -> { sx = rx; sy = ry }
                    }
                    val o = sy * rowStride + sx * 4
                    fb.put(i, (buf.get(o).toInt() and 0xFF) * k)
                    fb.put(plane + i, (buf.get(o + 1).toInt() and 0xFF) * k)
                    fb.put(2 * plane + i, (buf.get(o + 2).toInt() and 0xFF) * k)
                }
            }
        }
    }

    // ---------------------------------------------------------------- фото

    /** Для загруженных фото. */
    @Synchronized
    fun detectBitmap(src: Bitmap): List<Detection> {
        val t0 = System.nanoTime()
        val rw = src.width
        val rh = src.height
        val ts = targetSize(rw, rh)
        ensureBuf(ts[0], ts[1])
        val scale = min(inW / rw.toFloat(), inH / rh.toFloat())
        val nw = (rw * scale).toInt().coerceAtLeast(1)
        val nh = (rh * scale).toInt().coerceAtLeast(1)
        val padX = (inW - nw) / 2
        val padY = (inH - nh) / 2
        val scaled = Bitmap.createScaledBitmap(src, nw, nh, true)
        val px = IntArray(nw * nh)
        scaled.getPixels(px, 0, nw, 0, 0, nw, nh)
        if (scaled !== src) scaled.recycle()
        val plane = inW * inH
        if (fast) {
            val f = fArr
            Arrays.fill(f, GRAY)
            val l = lut
            val p2 = plane * 2
            for (y in 0 until nh) {
                val base = (padY + y) * inW + padX
                val sb = y * nw
                for (x in 0 until nw) {
                    val p = px[sb + x]
                    val i = base + x
                    f[i] = l[(p shr 16) and 0xFF]
                    f[plane + i] = l[(p shr 8) and 0xFF]
                    f[p2 + i] = l[p and 0xFF]
                }
            }
            inputBuf.clear()
            inputBuf.put(f)
            inputBuf.rewind()
            tabKey = "" // заливка серым перезапишется при следующем кадре камеры
        } else {
            val gray = GRAY
            val k = 1f / 255f
            val fb = inputBuf
            for (dy in 0 until inH) {
                for (dx in 0 until inW) {
                    val i = dy * inW + dx
                    val x = dx - padX
                    val y = dy - padY
                    if (x < 0 || y < 0 || x >= nw || y >= nh) {
                        fb.put(i, gray); fb.put(plane + i, gray); fb.put(2 * plane + i, gray)
                    } else {
                        val p = px[y * nw + x]
                        fb.put(i, ((p shr 16) and 0xFF) * k)
                        fb.put(plane + i, ((p shr 8) and 0xFF) * k)
                        fb.put(2 * plane + i, (p and 0xFF) * k)
                    }
                }
            }
        }
        preMs = (System.nanoTime() - t0) / 1e6f
        return infer(rw, rh, scale, padX, padY)
    }

    // ---------------------------------------------------------------- инференс

    private fun infer(rw: Int, rh: Int, scale: Float, padX: Int, padY: Int): List<Detection> {
        val t1 = System.nanoTime()
        inputBuf.rewind()
        val tensor = OnnxTensor.createTensor(env, inputBuf, longArrayOf(1, 3, inH.toLong(), inW.toLong()))
        val result = session.run(mapOf(inputName to tensor))
        val t2 = System.nanoTime()
        val raw: List<FloatArray> = try {
            val out = result[0] as OnnxTensor
            if (fast) parseFast(out) else parseClassic(out)
        } finally {
            result.close()
            tensor.close()
        }
        val dets = raw.map { r ->
            val l = ((r[0] - padX) / scale).coerceIn(0f, rw.toFloat()) / rw
            val t = ((r[1] - padY) / scale).coerceIn(0f, rh.toFloat()) / rh
            val rr = ((r[2] - padX) / scale).coerceIn(0f, rw.toFloat()) / rw
            val b = ((r[3] - padY) / scale).coerceIn(0f, rh.toFloat()) / rh
            val cls = r[5].toInt()
            Detection(l, t, rr, b, cls, names.getOrElse(cls) { "class$cls" }, r[4])
        }.filter { it.right > it.left && it.bottom > it.top }
        infMs = (t2 - t1) / 1e6f
        postMs = (System.nanoTime() - t2) / 1e6f
        return dets
    }

    /** Старый разбор: вывод целиком копируется во вложенные массивы. */
    private fun parseClassic(out: OnnxTensor): List<FloatArray> {
        val shape = out.info.shape
        val conf = conf
        val allowed = allowed
        @Suppress("UNCHECKED_CAST")
        val arr = (out.value as Array<Array<FloatArray>>)[0]
        val raw = ArrayList<FloatArray>()
        if (shape.size == 3 && shape[2] == 6L && shape[1] != 6L) {
            // YOLO26 end-to-end: [300, 6] = x1,y1,x2,y2,conf,cls (NMS не нужен)
            for (r in arr) {
                if (r[4] < conf) continue
                val cls = r[5].toInt()
                if (allowed.isNotEmpty() && cls !in allowed) continue
                raw.add(floatArrayOf(r[0], r[1], r[2], r[3], r[4], cls.toFloat()))
            }
        } else {
            // классический формат [4+nc, N] или [N, 4+nc]
            val transposed = shape[1] < shape[2]
            val n = if (transposed) arr[0].size else arr.size
            val ch = if (transposed) arr.size else arr[0].size
            val cand = ArrayList<FloatArray>()
            val classes: IntArray =
                if (allowed.isEmpty()) IntArray(ch - 4) { it } else allowed.filter { it < ch - 4 }.toIntArray()
            for (i in 0 until n) {
                var best = -1
                var bestS = 0f
                for (cls in classes) {
                    val s = if (transposed) arr[4 + cls][i] else arr[i][4 + cls]
                    if (s > bestS) { bestS = s; best = cls }
                }
                if (best < 0 || bestS < conf) continue
                val cx = if (transposed) arr[0][i] else arr[i][0]
                val cy = if (transposed) arr[1][i] else arr[i][1]
                val bw = if (transposed) arr[2][i] else arr[i][2]
                val bh = if (transposed) arr[3][i] else arr[i][3]
                cand.add(floatArrayOf(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, bestS, best.toFloat()))
            }
            raw.addAll(nms(cand, iou, 100000))
        }
        return if (raw.size > maxDet) raw.sortedByDescending { it[4] }.take(maxDet) else raw
    }

    /** Новый разбор: плоский массив без аллокаций на каждый анкер, последовательный доступ к памяти. */
    private fun parseFast(out: OnnxTensor): List<FloatArray> {
        val shape = out.info.shape
        if (shape.size != 3) return emptyList()
        val fb = out.floatBuffer
        val total = fb.remaining()
        if (outArr.size < total) outArr = FloatArray(total)
        fb.get(outArr, 0, total)
        val a = outArr
        val c = conf
        val al = allowed
        val md = maxDet

        if (shape[2] == 6L && shape[1] != 6L) {
            // end-to-end [N, 6]
            val rows = shape[1].toInt()
            val res = ArrayList<FloatArray>()
            for (i in 0 until rows) {
                val o = i * 6
                val sc = a[o + 4]
                if (sc < c) continue
                val cls = a[o + 5].toInt()
                if (al.isNotEmpty() && cls !in al) continue
                res.add(floatArrayOf(a[o], a[o + 1], a[o + 2], a[o + 3], sc, cls.toFloat()))
            }
            return if (res.size > md) res.sortedByDescending { it[4] }.take(md) else res
        }

        val d1 = shape[1].toInt()
        val d2 = shape[2].toInt()
        val transposed = d1 < d2          // [4+nc, N]
        val n = if (transposed) d2 else d1
        val ch = if (transposed) d1 else d2
        val obj = metaCount > 0 && ch == metaCount + 5 // YOLOv5-стиль: есть objectness
        val coff = if (obj) 5 else 4
        val ncls = ch - coff
        if (ncls <= 0 || n <= 0) return emptyList()
        val classes: IntArray =
            if (al.isEmpty()) IntArray(ncls) { it } else al.filter { it in 0 until ncls }.toIntArray()
        if (classes.isEmpty()) return emptyList()

        if (bestS.size < n) { bestS = FloatArray(n); bestC = IntArray(n) }
        val bs = bestS
        val bc = bestC
        Arrays.fill(bs, 0, n, c - 1e-6f)
        Arrays.fill(bc, 0, n, -1)

        if (transposed) {
            // идём по классам снаружи — память читается подряд
            for (cls in classes) {
                val base = (coff + cls) * n
                if (obj) {
                    val ob = 4 * n
                    for (i in 0 until n) {
                        val s = a[base + i] * a[ob + i]
                        if (s > bs[i]) { bs[i] = s; bc[i] = cls }
                    }
                } else {
                    for (i in 0 until n) {
                        val s = a[base + i]
                        if (s > bs[i]) { bs[i] = s; bc[i] = cls }
                    }
                }
            }
        } else {
            for (i in 0 until n) {
                val row = i * ch
                val ob = if (obj) a[row + 4] else 1f
                for (cls in classes) {
                    val s = a[row + coff + cls] * ob
                    if (s > bs[i]) { bs[i] = s; bc[i] = cls }
                }
            }
        }

        val cand = ArrayList<FloatArray>()
        for (i in 0 until n) {
            if (bc[i] < 0) continue
            val cx: Float
            val cy: Float
            val bw: Float
            val bh: Float
            if (transposed) {
                cx = a[i]; cy = a[n + i]; bw = a[2 * n + i]; bh = a[3 * n + i]
            } else {
                val row = i * ch
                cx = a[row]; cy = a[row + 1]; bw = a[row + 2]; bh = a[row + 3]
            }
            cand.add(floatArrayOf(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, bs[i], bc[i].toFloat()))
        }
        val res = nms(cand, iou, 400)
        return if (res.size > md) res.take(md) else res
    }

    private fun nms(boxes: List<FloatArray>, iouThr: Float, cap: Int): List<FloatArray> {
        var sorted = boxes.sortedByDescending { it[4] }
        if (sorted.size > cap) sorted = sorted.subList(0, cap)
        val keep = ArrayList<FloatArray>()
        for (b in sorted) {
            var ok = true
            for (k in keep) {
                if (k[5] == b[5] && iou(k, b) > iouThr) { ok = false; break }
            }
            if (ok) keep.add(b)
        }
        return keep
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val iw = min(a[2], b[2]) - max(a[0], b[0])
        val ih = min(a[3], b[3]) - max(a[1], b[1])
        if (iw <= 0 || ih <= 0) return 0f
        val inter = iw * ih
        val ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua
    }

    companion object {
        private const val GRAY = 114f / 255f

        val COCO = listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        )
    }
}
