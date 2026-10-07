package com.qaz1sm.yolodetect

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/** Рисует рамки поверх изображения, вписанного в View по принципу fitCenter. */
class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var imgW = 1
    private var imgH = 1
    private var mirror = false

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 38f
        isFakeBoldText = true
    }

    fun setResults(list: List<Detection>, imageW: Int, imageH: Int, mirrorX: Boolean) {
        detections = list
        imgW = imageW
        imgH = imageH
        mirror = mirrorX
        postInvalidate()
    }

    fun clear() {
        detections = emptyList()
        postInvalidate()
    }

    /** Область, которую занимает изображение внутри View. */
    fun imageRect(): RectF {
        val s = min(width / imgW.toFloat(), height / imgH.toFloat())
        val w = imgW * s
        val h = imgH * s
        val l = (width - w) / 2f
        val t = (height - h) / 2f
        return RectF(l, t, l + w, t + h)
    }

    private fun colorFor(id: Int): Int =
        Color.HSVToColor(floatArrayOf((id * 47 % 360).toFloat(), 0.85f, 1f))

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return
        val r = imageRect()
        val rw = r.width()
        val rh = r.height()
        for (d in detections) {
            var l = d.left
            var rt = d.right
            if (mirror) { l = 1f - d.right; rt = 1f - d.left }
            val box = RectF(r.left + l * rw, r.top + d.top * rh, r.left + rt * rw, r.top + d.bottom * rh)
            val col = colorFor(d.classId)
            boxPaint.color = col
            canvas.drawRect(box, boxPaint)
            val label = "${d.label} ${(d.score * 100).toInt()}%"
            val tw = textPaint.measureText(label)
            val th = textPaint.textSize
            val top = if (box.top - th - 10 < r.top) box.top else box.top - th - 10
            fillPaint.color = col
            canvas.drawRect(box.left, top, box.left + tw + 16, top + th + 10, fillPaint)
            textPaint.color = Color.BLACK
            canvas.drawText(label, box.left + 8, top + th - 4, textPaint)
        }
    }

    /** Применяется для сохранения: рисует рамки прямо на битмапе. */
    companion object {
        fun drawOnBitmap(src: android.graphics.Bitmap, dets: List<Detection>): android.graphics.Bitmap {
            val bmp = src.copy(android.graphics.Bitmap.Config.ARGB_8888, true)
            val c = Canvas(bmp)
            val k = maxOf(bmp.width, bmp.height) / 1000f
            val bp = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 4f * k }
            val fp = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
            val tp = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.BLACK; textSize = 30f * k; isFakeBoldText = true
            }
            for (d in dets) {
                val col = Color.HSVToColor(floatArrayOf((d.classId * 47 % 360).toFloat(), 0.85f, 1f))
                val l = d.left * bmp.width; val t = d.top * bmp.height
                val r = d.right * bmp.width; val b = d.bottom * bmp.height
                bp.color = col
                c.drawRect(l, t, r, b, bp)
                val label = "${d.label} ${(d.score * 100).toInt()}%"
                val tw = tp.measureText(label)
                val ty = if (t - tp.textSize - 8 * k < 0) t else t - tp.textSize - 8 * k
                fp.color = col
                c.drawRect(l, ty, l + tw + 12 * k, ty + tp.textSize + 8 * k, fp)
                c.drawText(label, l + 6 * k, ty + tp.textSize - 2 * k, tp)
            }
            return bmp
        }
    }
}
