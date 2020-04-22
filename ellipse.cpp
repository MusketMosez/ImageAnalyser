#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/videoio.hpp>

#include <stdio.h>
#include <math.h>

#include <iostream>

#include <cmath>

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>
#include <cmath>
#include <X11/Xlib.h>

using namespace cv;
using namespace std;
using namespace std::chrono ;

#include <pthread.h>

bool destroy=false;
bool go = false;
CvRect myBox;
bool drawing_box = false;

FILE *fpout;


//overloaded function to print RotatedRect object
ostream& operator<<(ostream& os, RotatedRect const& r)
{
  os << r.angle << "," << r.center << "," << r.size;
  return os;
}



float calcDistance(Point p1, Point p2){

  return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y,2));
}



float calcRatio(float d1, float d2){

  if (d1 > d2) {

    return d1/d2;     

    }

  else {
    return d2/d1;

      }
}


//function to draw principal components
Point2f drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{


  double angle = atan2( (double) p.y - q.y, (double) p.x - q.x ) ; // angle in radians
  double hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));

  cout << "Angle in radians " << angle << endl;
  cout << "Centre x :" << p.x << " Centre y :" << p.y << endl;
  cout << "Point x :" << q.x << " Point y :" << q.y << endl;

  //Lengthen the arrow by a factor of scale
  q.x = (int) (p.x - scale * hypotenuse * cos(angle));
  q.y = (int) (p.y - scale * hypotenuse * sin(angle));
  line(img, p, q, colour, 1, LINE_AA);

  Point2f stretched_point = q;

  // create the arrow hooks
  p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
  p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
  line(img, p, q, colour, 1, LINE_AA);

  p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
  p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
  line(img, p, q, colour, 1, LINE_AA);

  return stretched_point;

}


vector<float> getOrientation(const vector<Point> &pts, Mat &img)
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }

    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);

    //Store the center of the object
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                      static_cast<int>(pca_analysis.mean.at<double>(0, 1)));

    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);

    for (int i = 0; i < 2; i++)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));

        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);	
    }

    // Draw the principal components
    circle(img, cntr, 3, Scalar(255, 0, 255), 0.5);

    Point p1 = cntr + 0.4 * Point(static_cast<float>(eigen_vecs[0].x * eigen_val[0]), static_cast<float>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.4 * Point(static_cast<float>(eigen_vecs[1].x * eigen_val[1]), static_cast<float>(eigen_vecs[1].y * eigen_val[1]));
    
    Point po1 =  drawAxis(img, cntr, p1, Scalar(0,255, 0), 9);
    Point po2  =  drawAxis(img, cntr, p2, Scalar(255, 255, 0), 9);

       
    //calculate and store distances between point and centre
    float d1 = calcDistance(cntr, po1);
    float d2 = calcDistance(cntr, po2);

    //calculate ratio between magnitudes of drawn components
    float ratio = calcRatio(d1,d2);


    //calculate angle between drawn components
    po1 = po1 - cntr;
    po2 = po2 - cntr;
    float dot = (po1.x * po2.x) + (po1.y * po2.y);
    float det = (po1.x * po2.y) - (po1.y * po2.x);
    float angle = atan2(det,dot)/ (float)CV_PI*180.0f;

    angle = roundf(angle * 100) / 100;
    ratio = roundf(ratio * 100) / 100;

    vector<float> fArray;
    fArray.push_back(angle);
    fArray.push_back(ratio);
    return fArray;
}


vector<RotatedRect> getEllipsePCA( cv::Mat& binaryImg, cv::Mat drawing) {
 
  //vector to store ellipse objects
  vector<RotatedRect> results;

  int count = countNonZero(binaryImg);
  if (count == 0) {
    cout << "getEllipsePCA() encountered 0 pixels in binary image!" << endl;
    return vector<RotatedRect>();
  }

  //convert to matrix that contains point coordinates as column vectors 
  Mat data(2, count, CV_32FC1);
  int dataColumnIndex = 0;

  for (int row = 0; row < binaryImg.rows; row++) {

    for (int col = 0; col < binaryImg.cols; col++) {
      if (binaryImg.at<unsigned char>(row, col) != 0) {
	  data.at<float>(0, dataColumnIndex) = (float) col; //x coordinate
	  data.at<float>(1, dataColumnIndex) = (float) (binaryImg.rows - row); //y coordinate
	  ++dataColumnIndex;
      }
    }
  }


  
  //PCA
  const int maxComponents = 2;
  PCA pca(data, Mat() /*mean*/, CV_PCA_DATA_AS_COL, maxComponents);

  //result is contained in pca.eigenvectors (as row vectors)
 
  //get angle of principal axis
  float dx = pca.eigenvectors.at<float>(0, 0);
  float dy = pca.eigenvectors.at<float>(0, 1);
  float angle = atan2f(dy, dx) / (float)CV_PI*180.0f;

  
 
  //find the bounding rectangle with the given angle, by rotating the contour around the mean so that it is up-right
  //easily finding the box then

  Point2f centre(pca.mean.at<float>(0,0), binaryImg.rows - pca.mean.at<float>(1,0));
  Mat rotationMatrix = getRotationMatrix2D(centre, -angle, 1);
  Mat rotationMatrixInverse = getRotationMatrix2D(centre, angle, 1);

  vector< vector<Point> > contours;
  vector<vector<Point> > toErase;
  
  findContours(binaryImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
 
   for (size_t i =0; i < contours.size(); i++)
    {
      //calculate area of each contour
      double area = contourArea(contours[i]);
 
      //store contours that are too big or small
      if (area < 10 || area > 250) {
	toErase.push_back(contours[i]);
      }

    }

   
  //remove contours of incorrect size
  for ( auto it = contours.begin(); it != contours.end(); )
    {
      if (find( toErase.begin(), toErase.end(), *it) != toErase.end() )
	{
	  it = contours.erase( it );
	}
      else
	{
	  ++it;
	}
    }
  

  //iterate through each contour and convert to RotatedRect object 
  if (contours.size() != 0) {

   for (int i = 0; i < contours.size(); i++) {

    
     //turn vector of points into matrix (with points as column vectors, with a 3rd row full of 1's,
     //i.e. points are converted to extended coors
    RotatedRect result;
    
    Mat contourMat(3, contours[0].size(), CV_64FC1);
    double* row0 = contourMat.ptr<double>(0);
    double* row1 = contourMat.ptr<double>(1);
    double* row2 = contourMat.ptr<double>(2);
    for (int j = 0; j < (int) contours[i].size(); j++){
      row0[j] = (double) (contours[i])[j].x;
      row1[j] = (double) (contours[i])[j].y;
     row2[j] = 1;
    }
  
    Mat uprightContour = rotationMatrix*contourMat;

    //get min/max in order to determine width and height
    double minX, minY, maxX, maxY;
    minMaxLoc(Mat(uprightContour, Rect(0, 0, contours[0].size(), 1)), &minX, &maxX); //get minimum/maximum of first row
    minMaxLoc(Mat(uprightContour, Rect(0, 1, contours[0].size(), 1)), &minY, &maxY); //get minimum/maximum of second row

    int minXi = cvFloor(minX);
    int minYi = cvFloor(minY);
    int maxXi = cvCeil(maxX);
    int maxYi = cvCeil(maxY);

    //fill result
    result.angle = angle;
    result.size.width = (float) (maxXi - minXi);
    result.size.height = (float) (maxYi - minYi);

    //Find the correct centre
    Mat correctCentreUpright(3, 1, CV_64FC1);
    correctCentreUpright.at<double>(0, 0) = maxX - result.size.width/2;
    correctCentreUpright.at<double>(1, 0) = maxY - result.size.height/2;
    correctCentreUpright.at<double>(2, 0) = 1;

    Mat correctCentreMat = rotationMatrixInverse*correctCentreUpright;
    Point correctCentre = Point(cvRound(correctCentreMat.at<double>(0,0)), cvRound(correctCentreMat.at<double>(1,0)));

    result.center = correctCentre;
    results.push_back(result);

   }
  }

    return results;

}


string type2str(int type)
{
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}


//drawn region on interest
void draw_box(cv::Mat img, CvRect rect)
{

	rectangle(img, cvPoint(myBox.x, myBox.y), cvPoint(myBox.x+myBox.width,myBox.y+myBox.height),
			cvScalar(0,0,255) ,0.5);
}


//write results to output
void writeResult(int frameNumber, double a, double b, double e )
{
   cout << "writeResult() start" << endl;
   double pi = 3.1415;
   double area = pi * a *b;

   fprintf(fpout,"%04d,%6.2f,%6.2f,%6.4f,%6.2f\n",frameNumber,a,b,e,area);

   printf(        "%04d,%6.2f,%6.2f,%6.4f,%6.2f\n",frameNumber,a,b,e,area);


   cout << "writeResult()  end" << endl;
}



cv::Mat process (cv::Mat frame, cv::Mat bg)
{

	double yOffsetDraw = 200;
	auto begin = high_resolution_clock::now() ;
	static int n = 0;
	++n;

	Mat gray;
	Mat thresh;
	Mat bggray;
	Mat clFrame;

	
	clFrame = frame.clone();

        //region of interest
	cv::Mat myFrame(clFrame, myBox);

	//convert image to grayscale
	cvtColor(myFrame, gray, COLOR_BGR2GRAY);
	myFrame = gray.clone();
	cv::medianBlur(gray,gray,5);



	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;

	minMaxLoc( myFrame, &minVal, &maxVal, &minLoc, &maxLoc );
	cout << "min val : " << minVal << endl;
	cout << "max val: " << maxVal << endl;

	int const max_BINARY_value = maxVal;

	int threshold_value = maxVal*0.4; //125;

	cout << "threshold_value: " << threshold_value << endl;

	cv::threshold( gray, thresh, threshold_value, 255, 0 );
	

	Mat drawing = frame;
	RNG rng(12345);

        //store contours to be drawn in ellipses
        vector<RotatedRect> ellipses = getEllipsePCA(thresh, drawing);

	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

        if(ellipses.size() == 0) { cout << "Frame with no contour :" << n << endl; }
	
	//iterate through every contour 
	for (int i = 0; i < ellipses.size(); i++) {
	  
	  double w = ellipses[i].size.width;
	  double h = ellipses[i].size.height;

	  cout << "ellipse: r.center.x: " << ellipses[i].center.x << endl;
	  cout << "ellipse: r.center.y: " << ellipses[i].center.y << endl;
	  cout << "ellipse: r.size.width: " << ellipses[i].size.width << endl;
	  cout << "ellipse: r.size.height: " << ellipses[i].size.height << endl;

	  cout << "myBox.x: " << myBox.x << endl;
	  cout << "myBox.y: " << myBox.y << endl;
	  cout << "myBox.width: " << myBox.width << endl;
	  cout << "myBox.height: " << myBox.height << endl;

	  double xoffset = myBox.x;
	  double yoffset = myBox.y;

	  double minw = 8;
	  double minh = 8;
	  
          //add offset to be drawn correctly
	  ellipses[i].center.x += xoffset;
	  ellipses[i].center.y += yoffset;
	
          //write width and height to file
	  double a = w/2.0d;
	  double b = h/2.0d;

	  if(b > a)
	    {
	      double t = a;
	      a = b;
	      b = t;
	    }

	  double r = a/b;

	  //eccentricity calculation
	  double e = sqrt(a*a - b*b) / a;
	  writeResult(n, a, b, e );

	  e = roundf(e * 100) /100;
	  r = roundf(r * 100) / 100;
	  char buf[32];

	  sprintf(buf,"eccentricity = %4.2f", e);
	  string eratio(buf);
	  char buf2[32];
	  sprintf(buf2,"frame %4d", n);
	  string fnums(buf2);
	  char buf3[32];
	  sprintf(buf3, "ratio = %4.2f", r);
	  string rb(buf3);



	  putText(drawing,  fnums , Point(400, 120+yOffsetDraw), 1, 1, Scalar(255, 255, 0));

	  color = Scalar(255, 0, 0);



          //if the area is too big move to next contour
	  if(ellipses[i].size.height * ellipses[i].size.width > 400) continue;

	  
	  //draw contour 
	   ellipse( drawing, ellipses[i], Scalar(255, 0, 0), .2, 8 );
	 
          //Perform PCA again but for each drawn contour to determine axis of rotation
	  //and principal components
	  
	  cvtColor(drawing, gray, COLOR_BGR2GRAY);
	  cv::medianBlur(gray,gray,5);

	  cv::threshold( gray, thresh, threshold_value, 255, 0 );

	  vector<vector<Point> > contours;
          findContours(thresh, contours,  CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);         

	  vector<Point> Max = contours[0];
          double MaxArea = contourArea(contours[i]);
	  
	  for (size_t i = 0; i < contours.size(); i++)
	   {


	     // Calculate the area of each contour
	     double area = contourArea(contours[i]);
	    
	     // Ignore contours that are too small or too large
	     if (area >  25 && 1e4 > area) {

	       if (area > MaxArea) {
		 MaxArea = area;
		 Max = contours[i];
	 
	       }
	    
	     }
	   }
	
	  vector<float> vals = getOrientation(Max, drawing);    

	  auto end = high_resolution_clock::now() ;

	  auto ticks = duration_cast<microseconds>(end-begin) ;


	  //time to process current frame
	  string s = to_string(ticks.count());
	  s.append(" [usec]");
	  putText(drawing,  s , Point(400, 60+yOffsetDraw), 1, 1, Scalar(255, 255, 0));


	  //options to display relevenat values of frames e.g. angle between principal components,
	  //eccentrcity, ratios between width/height of ellipses and magnitudes of drawn components

	  eratio = to_string(e);
	  eratio.append(" E :");

	  // putText(drawing,  eratio , Point(400, 80+yOffsetDraw), 1, 1, Scalar(255, 255, 0));
	

	  rb = to_string(r);
	  rb.append(" W/h :");

	  putText(drawing,  rb , Point(400, 100+yOffsetDraw), 1, 1, Scalar(255, 255, 0));	

	  char buf4[6];
	  char buf5[6];

	  string an(buf4);
	  string pcr(buf5);
	   
	  float angle =vals[0];
	  float pcrat = vals[1];

	  an = to_string(angle);
	  an.append(" Angle ");

	  pcr = to_string(pcrat);
	  pcr.append( " Ratio ");

	  putText(drawing,  an , Point(400, 80+yOffsetDraw), 1, 1, Scalar(255, 255, 0));
	  // putText(drawing,  pcr , Point(400, 100+yOffsetDraw), 1, 1, Scalar(255, 255, 0));

	   
	   

	  cout << "process: duration = " << ticks.count() << " usec" << endl;
	 }
	
	cv::imshow("gray", gray);
	cv::imshow("canny", thresh);

	return frame;
}




void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if  ( event == EVENT_LBUTTONDOWN )
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if  ( event == EVENT_RBUTTONDOWN )
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if  ( event == EVENT_MBUTTONDOWN )
	{
		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if ( event == EVENT_MOUSEMOVE )
	{
		cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
	}
}

void mouse_pt_callback( int event, int x, int y, int flags, void* userdata )
{

	if(event == CV_EVENT_LBUTTONDOWN)
	{
		cv::Point* ptPtr = (cv::Point*)userdata;
		ptPtr->x = x;
		ptPtr->y = y;
	}

}


void start_processing()
{
	go = true;
	destroy=true;
}

void start_button_callback( int flags, void* userdata)
{
	start_processing();
}


void mouse_roi_callback( int event, int x, int y, int flags, void* userdata )
{

	switch( event )
	{
	case CV_EVENT_MOUSEMOVE:
	{
		if( drawing_box )
		{
			myBox.width = x-myBox.x;
			myBox.height = y-myBox.y;
		}
	}
	break;

	case CV_EVENT_LBUTTONDOWN:
	{
		drawing_box = true;
		myBox = cvRect( x, y, 0, 0 );
	}
	break;

	case CV_EVENT_LBUTTONUP:
	{
		drawing_box = false;
		if( myBox.width < 0 )
		{
			myBox.x += myBox.width;
			myBox.width *= -1;
		}

		if( myBox.height < 0 )
		{
			myBox.y += myBox.height;
			myBox.height *= -1;
		}
	}
	break;

	case CV_EVENT_RBUTTONUP:
	{
		go = true;
		destroy=true;
	}
	break;

	default:
		break;
	}

}


int main(int argc, char* argv[])
{
        //change directory as required
	fpout = fopen("/home/nathanmoses/workspace/Presentation/video/out/out.dat","w");

	fprintf(fpout,"frameNumber,ellipse,a,b,e,area\n");


	const char* name = "Find Ellipse";

	

	cv::VideoCapture cap(argv[1]);

	if (!cap.isOpened())
	{
		std::cout << "!!! Failed to open file: " << argv[1] << std::endl;
		return -1;
	}



	cv::Mat firstFrame;

	cv::Mat frame;

	cv::Mat roiFrame;

	cv::Mat myFrame;

	myBox = cvRect(0,0,1,1);

	int noFrames = 0;

	if (!cap.read(frame))
	{
		cout << "error cap.read(frame)" << endl;
		exit(1);
	}



	int width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));


	cout << "width = " << width <<", height = " << height <<endl;

	double FPS = cap.get(CV_CAP_PROP_FPS);

	cv::VideoWriter out("/home/nathanmoses/workspace/Presentation/video/out/output.avi", CV_FOURCC('M','J', 'P', 'G'), FPS, cv::Size(width, height));


	if(!out.isOpened())
	{
		std::cout <<"Error! Unable to open video file for output." << std::endl;
		std::exit(-2);
	}
	/*	 */

	firstFrame = frame.clone();

	cvNamedWindow(name,CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	cv::moveWindow 	( name, 100, 100);



namedWindow("gray",1);

namedWindow("canny",1);

	roiFrame = frame.clone();

	cv::Point2i pt(-1,-1);
	setMouseCallback(name, mouse_roi_callback, (void*)&roiFrame);
	
	while(1)
	{


		Mat aFrame = roiFrame.clone();


		draw_box(aFrame, myBox);

		cv::imshow(name, aFrame);

		char key = cvWaitKey(10);
		if (key == 27 || go)
		{
			go = false;
			break;
		}

	}


	auto begin_total = high_resolution_clock::now() ;

	for(;;)
	{

		auto begin = high_resolution_clock::now() ;

		if (!cap.read(frame))
			break;

		auto end = high_resolution_clock::now() ;

		auto ticks = duration_cast<microseconds>(end-begin) ;
		cout << "noFrame = " << noFrames << ", duration = "<< ticks.count() << " usec" << endl;


		myFrame = process (frame, firstFrame);


		draw_box(myFrame, myBox);



		cv::imshow(name, myFrame);

		string ty =  type2str(myFrame.type() );

		printf("src_cp: %s %dx%d \n", ty.c_str(),myFrame.cols, myFrame.rows );


	    char filename[PATH_MAX];

	    char* dir = "/home/nathanmoses/workspace/Presentation/video/out";

		{
                     sprintf(filename,"%simage%05d.png",dir,noFrames);

		     cv::imwrite(filename, frame);
		      
		}

		noFrames++;


		char key = cvWaitKey(1);
		if (key == 27) // ESC
			break;

	}


	auto end_total = high_resolution_clock::now() ;

	auto ticks_total = duration_cast<microseconds>(end_total-begin_total) ;
	cout << "noFrame = " << noFrames << ", duration = "<< ticks_total.count() << " usec" << endl;

	fclose(fpout);

	return 0;
}
