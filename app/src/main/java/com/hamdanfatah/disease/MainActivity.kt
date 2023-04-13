package com.hamdanfatah.disease

import android.content.Intent
import android.content.pm.PackageManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.Manifest
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Build
import com.hamdanfatah.disease.ml.DiseaseDetection
import org.checkerframework.checker.units.qual.min
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min


class MainActivity : AppCompatActivity() {

    lateinit var result: TextView
    lateinit var demoTxt: TextView
    lateinit var classified: TextView
    lateinit var clickHere: TextView
    lateinit var imageView: ImageView
    lateinit var arrowImage: ImageView
    lateinit var picture: Button

    val imageSize = 224
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        result = findViewById(R.id.result)
        imageView = findViewById(R.id.imageView)
        picture = findViewById(R.id.Button)

        demoTxt = findViewById(R.id.demoText)
        clickHere = findViewById(R.id.click_here)
        arrowImage = findViewById(R.id.demoArrow)
        classified = findViewById(R.id.classified)

        demoTxt.visibility = View.VISIBLE
        clickHere.visibility = View.GONE
        arrowImage.visibility = View.VISIBLE
        arrowImage.visibility = View.GONE
        classified.visibility = View.GONE
        result.visibility = View.GONE

        picture.setOnClickListener {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, 1)
            } else {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
                } else {
                    // Handle the case where the device is running an older version of Android
                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
                val image = data?.extras?.get("data") as? Bitmap
                val dimension = min(image?.width ?: 0, image?.height ?: 0)
                val thumbnail = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
                imageView.setImageBitmap(thumbnail)

                demoTxt.visibility = View.VISIBLE
                clickHere.visibility = View.GONE
                arrowImage.visibility = View.GONE
                classified.visibility = View.GONE
                result.visibility = View.GONE

                val scaledImage = Bitmap.createScaledBitmap(thumbnail, imageSize, imageSize, false)
                classifyImage(scaledImage)
            }

            super.onActivityResult(requestCode, resultCode, data)
    }

    private fun classifyImage(image: Bitmap) {
            try {
                val model = DiseaseDetection.newInstance(applicationContext)
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
                val byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
                byteBuffer.order(ByteOrder.nativeOrder())

                val intValues = IntArray(image.width * image.height)
                image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)

                var pixel = 0
                for (i in 0 until 224) {
                    for (j in 0 until 224) {
                        val value = intValues[pixel++]
                        byteBuffer.putFloat(((value shr 16) and 0xFF) * (1f / 255f))
                        byteBuffer.putFloat(((value shr 8) and 0xFF) * (1f / 255f))
                        byteBuffer.putFloat((value and 0xFF) * (1f / 255f))
                    }
                }
                inputFeature0.loadBuffer(byteBuffer)

                val output = model.process(inputFeature0)
                val outputFeature0 = output.outputFeature0AsTensorBuffer
                val confidence = outputFeature0.floatArray

                var maxPos = 0
                var maxConfidence = 0f
                for (i in confidence.indices) {
                    if (confidence[i] > maxConfidence) {
                        maxConfidence = confidence[i]
                        maxPos = i
                    }
                }

                val classes = arrayOf("Tomato healthy", "Tomato Spider mites Two spotted spider mite", "Tomato Tomato mosaic virus", "Tomato Tomato YellowLeaf Curl Virus", "Tomato Bacteria spot", "Tomato Septoria leaf spot")
                result.text = classes[maxPos]
                result.setOnClickListener {
                    startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.google.com/search?q=" + result.text)))
                }

                model.close()
            } catch (e: IOException) {
                // TODO Handle the exception
            }
        }







}