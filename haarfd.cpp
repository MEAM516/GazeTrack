 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"

 #include <iostream>
 #include <stdio.h>
 #include <math.h>

 using namespace std;
 using namespace cv;

 /** Function Headers */
 void detectAndDisplay( Mat frame );

 /** Global variables */
 String face_cascade_name = "haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 string window_name = "Capture - Face detection";
 RNG rng(12345);

 /** @function main */
 int main( int argc, const char** argv )
 {
	VideoCapture cap(0); // Open the Webcam
	if(!cap.isOpened())  // check if we succeeded
	return -1; 

	Mat frame;
   //-- 1. Load the cascades
   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   //if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };



     while( true )
     {
       cap >> frame;

   //-- 3. Apply the classifier to the frame
       if( !frame.empty() )
       { detectAndDisplay( frame ); }
       else
       { printf(" --(!) No captured frame -- Break!"); break; }

       int c = waitKey(10);
       if( (char)c == 'c' ) { break; }
      }

   return 0;
 }

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;
  Mat frametemp;
  double alpha;


   cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
   equalizeHist( frame_gray, frame_gray );

   //-- Detect faces
   face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
   cvtColor(frame_gray,frametemp,COLOR_GRAY2BGR);

   cout << "faces" <<faces.size() <<endl;

   for( size_t i = 0; i < faces.size(); i++ )
   {
     Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
	 Point point1(faces[i].x,faces[i].y);
	 Point point2(faces[i].x + faces[i].width,faces[i].y + faces[i].height);
     //rectangle(frame, point1, point2, Scalar( 0, 255, 0 ), 4, 8, 0 );
     //rectangle(frame_gray, point1, point2, Scalar( 0, 255, 0 ), 4, 8, 0 );
     //rectangle(frametemp, point1, point2, Scalar( 0, 255, 0 ), 4, 8, 0 );
     cout<<faces[i]<<endl;
     //cout<<<<endl;
     //Size size = frame_gray.size();
     //Size size2 = frame.size();
     //cout<<size.width<<size.height<<endl;
	 //cout<<size2.width<<size2.height<<endl;

     //Mat faceROI = frame( faces[i] );
     //Mat facegreyROI = frametemp(faces[i]);
     //Size size = faceROI.size();
     //cout<<size.width<<size.height<<endl;


     //frame.convertTo(frame,-1,3.0,0.0);


     // for( int y = 0; y < faceROI.rows; y++ )
     // { for( int x = 0; x < faceROI.cols; x++ )
     //    { //alpha = pow(((double)abs((x-faceROI.cols/2)*(y-faceROI.rows/2))/(faceROI.rows*faceROI.cols/4)),0.5);
     //    	alpha = exp(-(pow((x-faceROI.cols/2.0),2)/(50*faceROI.cols)+pow((y-faceROI.rows/2.0),2)/(50*faceROI.rows)));
     //    	for( int c = 0; c < 3; c++ )
     //         { faceROI.at<Vec3b>(y,x)[c] =
     //                     saturate_cast<uchar>( (alpha)*( faceROI.at<Vec3b>(y,x)[c] )+(1-alpha)*facegreyROI.at<Vec3b>(y,x)[c]); }
     //    }
     // }

     for( int y = 0; y < frame.rows; y++ )
     { for( int x = 0; x < frame.cols; x++ )
        { //alpha = pow(((double)abs((x-faceROI.cols/2)*(y-faceROI.rows/2))/(faceROI.rows*faceROI.cols/4)),0.5);
        	alpha = exp(-(pow((x-center.x),2)/(50*faces[i].width)+pow((y-center.y),2)/(50*faces[i].height)));
        	for( int c = 0; c < 3; c++ )
             { frametemp.at<Vec3b>(y,x)[c] =
                         saturate_cast<uchar>( (alpha)*( 3.0*frame.at<Vec3b>(y,x)[c] )+(1-alpha)*frametemp.at<Vec3b>(y,x)[c]); }
        }
     }



  //    cv::Rect roi( point1, cv::Size( faces[i].width, faces[i].height ));
	 // cv::Mat destinationROI = frametemp( roi );
	 // addWeighted(faceROI, 1, frametemp(faces[i]), 0, 0, faceROI);
	 // faceROI.copyTo( destinationROI );

     //frametemp(faces[i]) = frame(faces[i]);
//     std::vector<Rect> eyes;

 //    //-- In each face, detect eyes
 //    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
 //    cout << "eyes" << eyes.size()<<endl;

 //    for( size_t j = 0; j < eyes.size(); j++ )
 //     {
 //       Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
 //       int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
 //       circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
 //     }
   }
  //-- Show what you got
  imshow( window_name, frametemp);
  //if (faces.size()>0)
  //	{  imshow( window_name, frame(faces[0]));}
 }
