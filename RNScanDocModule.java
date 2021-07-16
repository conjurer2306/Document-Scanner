import android.app.Activity;
import android.content.ContentResolver;
import android.graphics.BitmapFactory;
import android.net.Uri;

import com.facebook.react.ReactInstanceManager;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.WritableNativeMap;

import android.graphics.Bitmap;
import android.provider.MediaStore;
import android.support.v7.appcompat.BuildConfig;
import android.util.Base64;
import android.widget.ImageView;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;

import java.io.IOException;

import static android.content.ContentValues.TAG;

class Line{
  Point _p1;
  Point _p2;
  Point _center;

  Line(Point p1, Point p2) {
    _p1 = p1;
    _p2 = p2;
    _center = new Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
  }
}

public class RNScanDocModule extends ReactContextBaseJavaModule {
  Bitmap srcBitmap;
  Bitmap grayBitmap;
  Bitmap cannyBitmap;
  Bitmap linesBitmap;
  Bitmap origBitmap;
  Bitmap dstBitmap;
  Bitmap outputBitmap;

  Mat rgbMat;
  Mat grayMat;
  Mat cannyMat;
  Mat linesMat;

  public final static String BASE64_PREFIX = "data:image/";
  public final static String CONTENT_PREFIX = "content://";
  public final static String FILE_PREFIX = "file:";

  private final ReactApplicationContext mReactContext;
  private final BaseLoaderCallback mOpenCVCallBack;

  public RNScanDocModule(ReactApplicationContext reactContext) {
    super(reactContext);
    this.mReactContext = reactContext;
    mOpenCVCallBack = new BaseLoaderCallback(getCurrentActivity()) {
      @Override
      public void onManagerConnected(int status) {
        switch (status) {
          case LoaderCallbackInterface.SUCCESS:
          {
            rgbMat = new Mat();
            grayMat = new Mat();
            cannyMat = new Mat();
            linesMat = new Mat();
            Log.d(TAG, "OpenCV loaded success");
          } break;
          default:
          {
            super.onManagerConnected(status);
          } break;
        }
      }
    };
  }

  @Override
  public String getName() {
    return "RNScanDoc";
  }

  @ReactMethod
  public void scan(String imagePath, int newWidth, int newHeight, int quality, String compressFormatString, String outputPath, Promise promise) {
    try {
      Log.d(TAG, "filepath: "+ imagePath);
      OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0,getCurrentActivity(),mOpenCVCallBack);

      Bitmap sourceImage;
      imagePath = imagePath.substring(6);

      if (!imagePath.startsWith(BASE64_PREFIX)) {
        Log.d(TAG, "not base64");
        sourceImage = loadBitmapFromFile(mReactContext, imagePath, newWidth, newHeight);
      }
      else {
        Log.d(TAG, "base64");
        sourceImage = loadBitmapFromBase64(imagePath);
      }

      Log.d(TAG, "finish load file");

      if (sourceImage == null) {
        promise.reject("1","Unable to load source image from path");
      }
      outputBitmap = findEdges(sourceImage);
      File path = mReactContext.getCacheDir();
      if (outputPath != null) {
        path = new File(outputPath);
      }

      Bitmap.CompressFormat compressFormat = Bitmap.CompressFormat.valueOf(compressFormatString);

      File file = saveImage(outputBitmap, path,
              Long.toString(new Date().getTime()), compressFormat, quality);

      WritableMap response = new WritableNativeMap();
      response.putString("path", Uri.fromFile(file).toString());
      promise.resolve(response);

    } catch (Exception e) {
      promise.reject("2",e.getMessage());
    }

  }
  private static File saveImage(Bitmap bitmap, File saveDirectory, String fileName,Bitmap.CompressFormat compressFormat, int quality) throws IOException {
    if (bitmap == null) {
      throw new IOException("The bitmap couldn't be resized");
    }

    File newFile = new File(saveDirectory, fileName + "." + compressFormat.name());
    if(!newFile.createNewFile()) {
      throw new IOException("The file already exists");
    }

    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    bitmap.compress(compressFormat, quality, outputStream);
    byte[] bitmapData = outputStream.toByteArray();

    outputStream.flush();
    outputStream.close();

    FileOutputStream fos = new FileOutputStream(newFile);
    fos.write(bitmapData);
    fos.flush();
    fos.close();

    return newFile;
  }

  private static Bitmap loadBitmapFromFile(Context context, String imagePath, int newWidth,int newHeight) throws IOException  {

    BitmapFactory.Options options = new BitmapFactory.Options();
    options.inJustDecodeBounds = true;
    loadBitmap(context, imagePath, options);
    Log.d(TAG, "loadBitmapFromFile: finish first load");

    options.inSampleSize = calculateInSampleSize(options, newWidth, newHeight);
    options.inJustDecodeBounds = false;
    return loadBitmap(context, imagePath, options);

  }

  private static int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
    final int height = options.outHeight;
    final int width = options.outWidth;

    int inSampleSize = 1;

    if (height > reqHeight || width > reqWidth) {
      final int halfHeight = height / 2;
      final int halfWidth = width / 2;

      while ((halfHeight / inSampleSize) >= reqHeight && (halfWidth / inSampleSize) >= reqWidth) {
        inSampleSize *= 2;
      }
    }

    return inSampleSize;
  }

  private static Bitmap loadBitmapFromBase64(String imagePath) {
    Bitmap sourceImage = null;

    final int prefixLen = BASE64_PREFIX.length();
    final boolean isJpeg = (imagePath.indexOf("jpeg") == prefixLen);
    final boolean isPng = (!isJpeg) && (imagePath.indexOf("png") == prefixLen);
    int commaLocation = -1;
    if (isJpeg || isPng){
      commaLocation = imagePath.indexOf(',');
    }
    if (commaLocation > 0) {
      final String encodedImage = imagePath.substring(commaLocation+1);
      final byte[] decodedString = Base64.decode(encodedImage, Base64.DEFAULT);
      sourceImage = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
    }

    return sourceImage;
  }

  private static Bitmap loadBitmap(Context context, String imagePath, BitmapFactory.Options options) throws IOException {
    Bitmap sourceImage = null;
    if (!imagePath.startsWith(CONTENT_PREFIX)) {
      try {
        Log.d(TAG, "loadBitmap: " + imagePath);
        sourceImage = BitmapFactory.decodeFile(imagePath, options);
      } catch (Exception e) {
        e.printStackTrace();
        throw new IOException("Error decoding image file");
      }
    } else {
      ContentResolver cr = context.getContentResolver();
      InputStream input = cr.openInputStream(Uri.parse(imagePath));
      if (input != null) {
        sourceImage = BitmapFactory.decodeStream(input, null, options);
        input.close();
      }
    }
    return sourceImage;
  }

  protected Bitmap findEdges(Bitmap bitmap) {

    BitmapFactory.Options o=new BitmapFactory.Options();
    o.inSampleSize = 4;
    o.inDither=false;
    origBitmap = bitmap;

    int w = origBitmap.getWidth();
    int h = origBitmap.getHeight();
    int min_w = 800;
    double scale = Math.min(10.0, w*1.0/ min_w);
    int w_proc = (int) (w * 1.0 / scale);
    int h_proc = (int) (h * 1.0 / scale);
    srcBitmap = Bitmap.createScaledBitmap(origBitmap, w_proc, h_proc, false);
    grayBitmap = Bitmap.createBitmap(w_proc, h_proc, Bitmap.Config.RGB_565);
    cannyBitmap = Bitmap.createBitmap(w_proc, h_proc, Bitmap.Config.RGB_565);
    linesBitmap = Bitmap.createBitmap(w_proc, h_proc, Bitmap.Config.RGB_565);


    Utils.bitmapToMat(srcBitmap, rgbMat);
    Imgproc.cvtColor(rgbMat, grayMat, Imgproc.COLOR_RGB2GRAY);
    cannyMat = getCanny(grayMat);
    Imgproc.HoughLinesP(cannyMat,linesMat, 1, Math.PI/180, w_proc/12, w_proc/12, 20 );
    Log.e("opencv","lines.cols " + linesMat.cols() + " w_proc/3: " + w_proc/3);
    List<Line> horizontals = new ArrayList<>();
    List<Line> verticals = new ArrayList<>();
    for (int x = 0; x < linesMat.rows(); x++)
    {
      double[] vec = linesMat.get(x, 0);
      double x1 = vec[0],
              y1 = vec[1],
              x2 = vec[2],
              y2 = vec[3];
      Point start = new Point(x1, y1);
      Point end = new Point(x2, y2);
      Line line = new Line(start, end);
      if (Math.abs(x1 - x2) > Math.abs(y1-y2)) {
        horizontals.add(line);
      } else {
        verticals.add(line);
      }
      if (BuildConfig.DEBUG) {
      }
    }

    Log.e("HoughLines","completed HoughLines");
    Log.e("HoughLines","linesMat size: " + linesMat.size());
    Log.e("HoughLines", "linesBitmap size: " + Integer.toString(linesBitmap.getHeight()) +" x " + Integer.toString(linesBitmap.getWidth()));
    Log.e("Lines Detected", Integer.toString(linesMat.rows()));

    if (linesMat.rows() > 400)
    {
      Context context = getReactApplicationContext();
      int duration = Toast.LENGTH_LONG;
      Toast toast = Toast.makeText(context, "Please use a cleaner background",duration);
      toast.show();
    }

    if (horizontals.size() < 2) {
      if (horizontals.size() == 0 || horizontals.get(0)._center.y > h_proc /2) {
        horizontals.add(new Line(new Point(0,0),new Point(w_proc-1, 0)));
      }
      if (horizontals.size() == 0 || horizontals.get(0)._center.y <= h_proc /2) {
        horizontals.add(new Line(new Point(0,h_proc-1),new Point(w_proc-1, h_proc-1)));
      }
    }
    if (verticals.size() < 2) {
      if (verticals.size() == 0 || verticals.get(0)._center.x > w_proc / 2) {
        verticals.add(new Line(new Point(0, 0), new Point(h_proc - 1, 0)));
      }
      if (verticals.size() == 0 || verticals.get(0)._center.x <= w_proc / 2) {
        verticals.add(new Line(new Point(w_proc - 1, 0), new Point(w_proc - 1, h_proc - 1)));
      }
    }

    Collections.sort(horizontals, new Comparator<Line>() {
      @Override
      public int compare(Line lhs, Line rhs) {
        return (int)(lhs._center.y - rhs._center.y);
      }
    });

    Collections.sort(verticals, new Comparator<Line>() {
      @Override
      public int compare(Line lhs, Line rhs) {
        return (int)(lhs._center.x - rhs._center.x);
      }
    });

    List<Point> intersections = new ArrayList<>();
    intersections.add(computeIntersection(horizontals.get(0),verticals.get(0)));
    intersections.add(computeIntersection(horizontals.get(0),verticals.get(verticals.size()-1)));
    intersections.add(computeIntersection(horizontals.get(horizontals.size()-1),verticals.get(0)));
    intersections.add(computeIntersection(horizontals.get(horizontals.size()-1),verticals.get(verticals.size()-1)));

    Log.e("Intersections", Double.toString(intersections.get(0).x));

    Log.e("Intersections", Double.toString(intersections.get(0).x));

    double w1 = Math.sqrt( Math.pow(intersections.get(3).x - intersections.get(2).x , 2) + Math.pow(intersections.get(3).x - intersections.get(2).x , 2));
    double w2 = Math.sqrt( Math.pow(intersections.get(1).x - intersections.get(0).x , 2) + Math.pow(intersections.get(1).x - intersections.get(0).x , 2));
    double h1 = Math.sqrt( Math.pow(intersections.get(1).y - intersections.get(3).y , 2) + Math.pow(intersections.get(1).y - intersections.get(3).y , 2));
    double h2 = Math.sqrt( Math.pow(intersections.get(0).y - intersections.get(2).y , 2) + Math.pow(intersections.get(0).y - intersections.get(2).y , 2));

    double maxWidth = (w1 < w2) ? w1 : w2;
    double maxHeight = (h1 < h2) ? h1 : h2;
    Mat srcMat = new Mat(4,1,CvType.CV_32FC2);
    srcMat.put(0,0,intersections.get(0).x,intersections.get(0).y,intersections.get(1).x,intersections.get(1).y,intersections.get(2).x,intersections.get(2).y,intersections.get(3).x,intersections.get(3).y);

    Mat dstMat = new Mat(4,1,CvType.CV_32FC2);
    dstMat.put(0,0, 0.0,0.0, maxWidth-1,0.0, 0.0,maxHeight-1, maxWidth-1, maxHeight-1);
    Log.e("FinalDisplay","srcMat: " + srcMat.size());
    Log.e("FinalDisplay","dstMat: " + dstMat.size());

    Mat transformMatrix = Imgproc.getPerspectiveTransform(srcMat,dstMat);

    Mat finalMat = Mat.zeros((int)maxHeight, (int)maxWidth ,CvType.CV_32FC2);
    Imgproc.warpPerspective(rgbMat, finalMat, transformMatrix, finalMat.size());
    Log.e("FinalDisplay","finalMat: " + finalMat.size());

    dstBitmap = Bitmap.createBitmap(finalMat.width(), finalMat.height(), Bitmap.Config.RGB_565);
    Log.e("FinalDisplay","dstBitmap: " + dstBitmap.getWidth() + " x " + dstBitmap.getHeight());
    Utils.matToBitmap(finalMat, dstBitmap);

    return dstBitmap;
  }

  protected Mat getCanny(Mat gray) {
    Mat threshold = new Mat();
    Mat canny = new Mat();

    double high_threshold = Imgproc.threshold(gray, threshold, 0, 255, 8);
    double low_threshold = high_threshold * 0.5;
    Imgproc.Canny(gray, canny, low_threshold, high_threshold);
    return canny;
  }

  protected Point computeIntersection (Line l1, Line l2) {
    double x1 = l1._p1.x, x2= l1._p2.x, y1 = l1._p1.y, y2 = l1._p2.y;
    double x3 = l2._p1.x, x4 = l2._p2.x, y3 = l2._p1.y, y4 = l2._p2.y;
    double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    Point pt = new Point();
    pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
    pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
    return pt;
  }
}