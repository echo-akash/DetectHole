package com.example.detecthole;

import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    public static final String TAG = MainActivity.class.getSimpleName();

    private ImageView mImageView;
    private Button mProcessButton;

    private Mat mSourceImageMat;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImageView = (ImageView) findViewById(R.id.target_image_view);
        mProcessButton = (Button) findViewById(R.id.process_button);
        mProcessButton.setVisibility(View.INVISIBLE);

        mProcessButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                processImage();
            }
        });
    }

    private void processImage() {
        try {
            mSourceImageMat = Utils.loadResource(this, R.drawable.target);
            Bitmap bm = Bitmap.createBitmap(mSourceImageMat.cols(), mSourceImageMat.rows(),Bitmap.Config.ARGB_8888);

            final Mat mat = new Mat();
            final List<Mat> channels = new ArrayList<>(3);

            mSourceImageMat.copyTo(mat);

            // split image channels: 0-H, 1-S, 2-V
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2HSV);
            Core.split(mat, channels);
            final Mat frameV = channels.get(2);

            // find white areas with max brightness
            Imgproc.threshold(frameV, frameV, 245, 255, Imgproc.THRESH_BINARY);

            // find contours
            List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
            Imgproc.findContours(frameV, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

            // find average contour area for "twin" hole detection
            double averageArea = 0;
            int contoursCount = 0;
            Iterator<MatOfPoint> each = contours.iterator();
            while (each.hasNext()) {
                averageArea += Imgproc.contourArea(each.next());
                contoursCount++;
            }
            if (contoursCount != 0) {
                averageArea /= contoursCount;
            }

            int holesCount = 0;
            each = contours.iterator();
            while (each.hasNext()) {
                MatOfPoint contour = each.next();

                MatOfPoint2f areaPoints = new MatOfPoint2f(contour.toArray());
                RotatedRect boundingRect = Imgproc.minAreaRect(areaPoints);
                Point rect_points[] = new Point[4];

                boundingRect.points(rect_points);
                for(int i=0; i<4; ++i){
                    Imgproc.line(mSourceImageMat, rect_points[i], rect_points[(i+1)%4], new Scalar(255,0,0), 2);
                }
                holesCount++;

                Imgproc.putText(mSourceImageMat, Integer.toString(holesCount), new Point(boundingRect.center.x + 20, boundingRect.center.y),
                        Core.FONT_HERSHEY_PLAIN, 1.5 ,new Scalar(255, 0, 0));

                // case of "twin" hole (like 9 & 10) on image
                if (Imgproc.contourArea(contour) > 1.3f * averageArea) {
                    holesCount++;
                    Imgproc.putText(mSourceImageMat, ", " + Integer.toString(holesCount), new Point(boundingRect.center.x + 40, boundingRect.center.y),
                            Core.FONT_HERSHEY_PLAIN, 1.5 ,new  Scalar(255, 0, 0));
                }

            }

            // convert to bitmap:
            Utils.matToBitmap(mSourceImageMat, bm);
            mImageView.setImageBitmap(bm);

            // release
            frameV.release();
            mat.release();

        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    @Override
    protected void onPostResume() {
        super.onPostResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mOpenCVLoaderCallback);
    }

    private BaseLoaderCallback mOpenCVLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mProcessButton.setVisibility(View.VISIBLE);
                } break;
                default: {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };



}
