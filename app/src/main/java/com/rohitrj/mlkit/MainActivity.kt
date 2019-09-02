package com.rohitrj.mlkit

import android.app.Activity
import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.util.Pair
import android.view.View
import android.widget.*
import com.google.android.gms.tasks.OnSuccessListener
import com.google.firebase.ml.common.FirebaseMLException
import com.google.firebase.ml.common.modeldownload.FirebaseLocalModel
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager
import com.google.firebase.ml.common.modeldownload.FirebaseRemoteModel
import com.google.firebase.ml.custom.*
import com.google.firebase.ml.vision.FirebaseVision
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.document.FirebaseVisionDocumentText
import com.google.firebase.ml.vision.face.FirebaseVisionFace
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions
import com.google.firebase.ml.vision.text.FirebaseVisionText
import com.google.firebase.ml.vision.text.FirebaseVisionTextRecognizer
import java.io.*
import java.lang.Exception
import java.lang.StringBuilder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.KeyStore
import java.util.*
import java.util.Map
import kotlin.Comparator
import kotlin.experimental.and

@Suppress("RECEIVER_NULLABILITY_MISMATCH_BASED_ON_JAVA_ANNOTATIONS")
class MainActivity : AppCompatActivity(), AdapterView.OnItemSelectedListener {

    val TAG = "MainActivity"
    lateinit var mImageView: ImageView
    lateinit var mTextButton: Button
    lateinit var mFaceButton: Button
    lateinit var mCloudButton: Button
    lateinit var mRunCustomModelButton: Button
    lateinit var mSelectedImage: Bitmap
    lateinit var mGraphicOverlay: GraphicOverlay
    // Max width (portrait mode)
    var mImageMaxWidth: Int? = null
    // Max height (portrait mode)
    var mImageMaxHeight: Int? = null

    /**
     * An instance of the driver class to run model inference with Firebase.
     */
    var mInterpreter: FirebaseModelInterpreter? = null
    /**
     * Data configuration of input & output data of model.
     */
    lateinit var mDataOptions: FirebaseModelInputOutputOptions

    /**
     * Name of the model file hosted with Firebase.
     */
    private val HOSTED_MODEL_NAME = "mobilenet_v1_224_quant"
    private val LOCAL_MODEL_ASSET = "mobilenet_v1_1.0_224_quant.tflite"
    /**
     * Name of the label file stored in Assets.
     */
    private val LABEL_PATH = "labels.txt"
    /**
     * Number of results to show in the UI.
     */
    private val RESULTS_TO_SHOW = 3
    /**
     * Dimensions of inputs.
     */
    private val DIM_BATCH_SIZE = 1
    private val DIM_PIXEL_SIZE = 3
    private val DIM_IMG_SIZE_X = 224
    private val DIM_IMG_SIZE_Y = 224
    /**
     * Labels corresponding to the output of the vision model.
     */
    private var mLabelList: List<String>? = null

    private val sortedLabels = PriorityQueue<kotlin.collections.Map.Entry<String, Float>>(
        RESULTS_TO_SHOW,
        Comparator<kotlin.collections.Map.Entry<String, Float>>
        { p0, p1 -> (p0!!.component2().compareTo(p1!!.component2())) }
        )
    /* Preallocated buffers for storing image data. */

    private val intValues = IntArray(DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y)


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        mImageView = findViewById(R.id.image_view)
        mTextButton = findViewById(R.id.button_text)
        mFaceButton = findViewById(R.id.button_face)
        mCloudButton = findViewById(R.id.button_cloud_text)
        mRunCustomModelButton = findViewById(R.id.button_run_custom_model)
        mGraphicOverlay = findViewById(R.id.graphic_overlay)

        mTextButton.setOnClickListener(View.OnClickListener { runTextRecognition() })
        mFaceButton.setOnClickListener(View.OnClickListener { runFaceContourDetection() })
        mCloudButton.setOnClickListener(View.OnClickListener { runCloudTextRecognition() })
        mRunCustomModelButton.setOnClickListener(View.OnClickListener { runModelInference() })
        val dropdown: Spinner = findViewById(R.id.spinner)
        val items = arrayOf(
            "Test Image 1 (Text)", "Test Image 2 (Text)",
            "Test Image 3" + " (Face)", "Test Image 4 (Object)", "Test Image 5 (Object)"
        )
        val adapter = ArrayAdapter(
            this, android.R.layout
                .simple_spinner_dropdown_item, items
        )
        dropdown.adapter = adapter
        dropdown.onItemSelectedListener = this
        initCustomModel()
    }

    private fun runTextRecognition() {
        val image = FirebaseVisionImage.fromBitmap(mSelectedImage)
        val recognizer = FirebaseVision.getInstance()
            .onDeviceTextRecognizer
        mTextButton.isEnabled = false
        recognizer.processImage(image).addOnSuccessListener {
            mTextButton.isEnabled = true
            processTextRecognitionResult(it)
        }.addOnFailureListener {
            mTextButton.isEnabled = true
            it.printStackTrace()
        }
    }

    private fun processTextRecognitionResult(texts: FirebaseVisionText) {
        mGraphicOverlay.clear()
        val blocks: List<FirebaseVisionText.TextBlock> = texts.textBlocks
        if (blocks.isEmpty()) {
            showToast("No text found")
            return
        }
        for (i in blocks) {
            val lines: List<FirebaseVisionText.Line> = i.lines
            for (j in lines) {
                val elements: List<FirebaseVisionText.Element> = j.elements
                for (k in elements) {
                    val textGraphic: GraphicOverlay.Graphic = TextGraphic(mGraphicOverlay, k)
                    mGraphicOverlay.add(textGraphic)

                }
            }
        }

    }

    private fun runFaceContourDetection() {
        val image: FirebaseVisionImage = FirebaseVisionImage.fromBitmap(mSelectedImage)
        val options: FirebaseVisionFaceDetectorOptions =
            FirebaseVisionFaceDetectorOptions.Builder()
                .setPerformanceMode(FirebaseVisionFaceDetectorOptions.FAST)
                .setContourMode(FirebaseVisionFaceDetectorOptions.ALL_CONTOURS)
                .build()

        mFaceButton.isEnabled = false
        val detector: FirebaseVisionFaceDetector = FirebaseVision.getInstance()
            .getVisionFaceDetector(options)
        detector.detectInImage(image)
            .addOnSuccessListener {
                mFaceButton.setEnabled(true)
                processFaceContourDetectionResult(it)
            }
            .addOnFailureListener {
                mFaceButton.setEnabled(true)
            }
    }

    private fun processFaceContourDetectionResult(faces: List<FirebaseVisionFace>) {
        // Task completed successfully
        mGraphicOverlay.clear()
        if (faces.isEmpty()) {
            showToast("No face found")
            return
        }
        for (i in faces) {
            val face: FirebaseVisionFace = i
            val faceGraphic = FaceContourGraphic(mGraphicOverlay)
            mGraphicOverlay.add(faceGraphic)
            faceGraphic.updateFace(face)
        }
    }

    private fun initCustomModel() {

        mLabelList = loadLabelList(this)
        val inputDims = intArrayOf(DIM_BATCH_SIZE, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, DIM_PIXEL_SIZE)
        val outputDims = intArrayOf(DIM_BATCH_SIZE, mLabelList!!.size)

        try {
            mDataOptions = FirebaseModelInputOutputOptions.Builder()
                .setInputFormat(0, FirebaseModelDataType.BYTE, inputDims)
                .setOutputFormat(0, FirebaseModelDataType.BYTE, outputDims)
                .build()

            val conditions: FirebaseModelDownloadConditions = FirebaseModelDownloadConditions
                .Builder()
                .build()

            val remoteModel: FirebaseRemoteModel = FirebaseRemoteModel.Builder(HOSTED_MODEL_NAME)
                .enableModelUpdates(true)
                .setInitialDownloadConditions(conditions)
                .setUpdatesDownloadConditions(conditions)  // You could also specify
                // different conditions
                // for updates
                .build()

            val localModel: FirebaseLocalModel = FirebaseLocalModel.Builder("asset")
                .setAssetFilePath(LOCAL_MODEL_ASSET).build()
            val manager: FirebaseModelManager = FirebaseModelManager.getInstance()
            manager.registerRemoteModel(remoteModel)
            manager.registerLocalModel(localModel)
            val modelOptions: FirebaseModelOptions = FirebaseModelOptions.Builder()
                .setRemoteModelName(HOSTED_MODEL_NAME)
                .setLocalModelName("asset")
                .build()
            mInterpreter = FirebaseModelInterpreter.getInstance(modelOptions)!!

        } catch (e: FirebaseMLException) {
            showToast("Error while setting up the model")
            e.printStackTrace()
        }

    }

    private fun runModelInference() {
        mGraphicOverlay.clear()
        if (mInterpreter == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.")
            return
        }
        // Create input data.
        val imgData: ByteBuffer = convertBitmapToByteBuffer(
            mSelectedImage, mSelectedImage.width,
            mSelectedImage.height
        )

        try {
            val inputs: FirebaseModelInputs = FirebaseModelInputs.Builder()
                .add(imgData).build()
            // Here's where the magic happens!!
            mInterpreter!!.run(inputs, mDataOptions)
                .addOnFailureListener {
                    it.printStackTrace()
                    showToast("Error running model inference")
                }
                .continueWith {
//                    Log.i(TAG,it.result!!.getOutput(0))
                    val labelProbArray: Array<ByteArray> = it.result!!.getOutput(0)
                    val topLabels: List<String> = getTopLabels(labelProbArray)
                    val labelGraphic: GraphicOverlay.Graphic =
                        LabelGraphic(mGraphicOverlay, topLabels)
                    mGraphicOverlay.add(labelGraphic)
                    return@continueWith topLabels
                }
        } catch (e: FirebaseMLException) {
            e.printStackTrace()
            showToast("Error  exception running model inference")
        }


    }

    private fun runCloudTextRecognition() {
        // Replace with code from the codelab to run cloud text recognition.
    }

    private fun processCloudTextRecognitionResult(text: FirebaseVisionDocumentText) {
        // Replace with code from the codelab to process the cloud text recognition result.
    }

    /**
     * Gets the top labels in the results.
     */
    @Synchronized
    private fun getTopLabels(labelProbArray: Array<ByteArray>): List<String> {
        if (mLabelList != null) {
            for (i in mLabelList!!.indices) {
                val prob = (labelProbArray[0][i] and 0xff.toByte())/255.0f
                Log.i(TAG, "${mLabelList!![i]} :: $prob")

                sortedLabels.add(
                    AbstractMap.SimpleEntry(
                        mLabelList!![i],
                        (labelProbArray[0][i] and 0xff.toByte()) / 255.0f
                    )
                )
                if (sortedLabels.size > RESULTS_TO_SHOW) {
                    sortedLabels.poll()
                }
            }
        }
        val result = ArrayList<String>()
        val size = sortedLabels.size
        for (i in 0 until size) {
            val label = sortedLabels.poll()
            result.add(label.key + ":" + label.value)
        }
        Log.i(TAG, "labels: $result")
        return result
    }

    /**
     * Reads label list from Assets.
     */
    private fun loadLabelList(activity: Activity): List<String> {
        val labelList = ArrayList<String>()
        val assetManager: AssetManager = resources.assets
        var inputStream: InputStream? = null
        try {
            inputStream = assetManager.open(LABEL_PATH)
            val s = Scanner(inputStream)
            while(s.hasNext()){
                labelList.add( s.nextLine() )
            }

        } catch (e: IOException) {
            e.printStackTrace()
        }
        return labelList
    }


    /**
     * Writes Image data into a `ByteBuffer`.
     */
    @Synchronized
    private fun convertBitmapToByteBuffer(
        bitmap: Bitmap, width: Int, height: Int
    ): ByteBuffer {
        val imgData = ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE
        )
        imgData.order(ByteOrder.nativeOrder())
        val scaledBitmap = Bitmap.createScaledBitmap(
            bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y,
            true
        )
        imgData.rewind()
        scaledBitmap.getPixels(
            intValues, 0, scaledBitmap.width, 0, 0,
            scaledBitmap.width, scaledBitmap.height
        )
        // Convert the image to int points.
        var pixel = 0
        for (i in 0 until DIM_IMG_SIZE_X) {
            for (j in 0 until DIM_IMG_SIZE_Y) {
                val `val` = intValues[pixel++]
                imgData.put((`val` shr 16 and 0xFF).toByte())
                imgData.put((`val` shr 8 and 0xFF).toByte())
                imgData.put((`val` and 0xFF).toByte())
            }
        }
        return imgData
    }

    private fun showToast(message: String) {
        Toast.makeText(applicationContext, message, Toast.LENGTH_SHORT).show()
    }

    // Functions for loading images from app assets.

    // Returns max image width, always for portrait mode. Caller needs to swap width / height for
    // landscape mode.
    private fun getImageMaxWidth(): Int? {
        if (mImageMaxWidth == null) {
            // Calculate the max width in portrait mode. This is done lazily since we need to
            // wait for
            // a UI layout pass to get the right values. So delay it to first time image
            // rendering time.
            mImageMaxWidth = mImageView.width
        }

        return mImageMaxWidth
    }

    // Returns max image height, always for portrait mode. Caller needs to swap width / height for
    // landscape mode.
    private fun getImageMaxHeight(): Int? {
        if (mImageMaxHeight == null) {
            // Calculate the max width in portrait mode. This is done lazily since we need to
            // wait for
            // a UI layout pass to get the right values. So delay it to first time image
            // rendering time.
            mImageMaxHeight = mImageView.height
        }

        return mImageMaxHeight
    }

    // Gets the targeted width / height.
    private fun getTargetedWidthHeight(): Pair<Int, Int> {
        val targetWidth: Int
        val targetHeight: Int
        val maxWidthForPortraitMode = getImageMaxWidth()!!
        val maxHeightForPortraitMode = getImageMaxHeight()!!
        targetWidth = maxWidthForPortraitMode
        targetHeight = maxHeightForPortraitMode
        return Pair(targetWidth, targetHeight)
    }

    override fun onItemSelected(parent: AdapterView<*>, v: View, position: Int, id: Long) {
//        mGraphicOverlay.clear()
        when (position) {
            0 -> mSelectedImage = getBitmapFromAsset(this, "Please_walk_on_the_grass.jpg")
            1 ->
                // Whatever you want to happen when the thrid item gets selected
                mSelectedImage = getBitmapFromAsset(this, "nl2.jpg")
            2 ->
                // Whatever you want to happen when the thrid item gets selected
                mSelectedImage = getBitmapFromAsset(this, "grace_hopper.jpg")
            3 ->
                // Whatever you want to happen when the thrid item gets selected
                mSelectedImage = getBitmapFromAsset(this, "tennis.jpg")
            4 ->
                // Whatever you want to happen when the thrid item gets selected
                mSelectedImage = getBitmapFromAsset(this, "mountain.jpg")
        }
        // Get the dimensions of the View
        val targetedSize = getTargetedWidthHeight()

        val targetWidth = targetedSize.first
        val maxHeight = targetedSize.second

        // Determine how much to scale down the image
        val scaleFactor = Math.max(
            mSelectedImage.width.toFloat() / targetWidth.toFloat(),
            mSelectedImage.height.toFloat() / maxHeight.toFloat()
        )

        val resizedBitmap = Bitmap.createScaledBitmap(
            mSelectedImage,
            (mSelectedImage.width / scaleFactor).toInt(),
            (mSelectedImage.height / scaleFactor).toInt(),
            true
        )

        mImageView.setImageBitmap(resizedBitmap)
        mSelectedImage = resizedBitmap
    }

    override fun onNothingSelected(parent: AdapterView<*>) {
        // Do nothing
    }

    fun getBitmapFromAsset(context: Context, filePath: String): Bitmap {
        val assetManager = context.assets

        val `is`: InputStream
        var bitmap: Bitmap? = null
        try {
            `is` = assetManager.open(filePath)
            bitmap = BitmapFactory.decodeStream(`is`)
        } catch (e: IOException) {
            e.printStackTrace()
        }

        return bitmap!!
    }
}
