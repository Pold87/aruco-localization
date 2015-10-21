#include <iostream>
#include <aruco/aruco.h>
#include <aruco/cvdrawingutils.h>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace aruco;
int main(int argc,char **argv)
{
    try
    {
        if (argc!=2) {
            cerr<<"Usage: in.jpg "<<endl;
            return -1;
        }
        MarkerDetector MDetector;
        vector<Marker> Markers;
        //read the input image
        cv::Mat InImage;
        InImage=cv::imread(argv[1]);
     //Ok, let's detect
        MDetector.detect(InImage,Markers);
        //for each marker, draw info and its boundaries in the image
        for (unsigned int i=0;i<Markers.size();i++) {
            cout<<Markers[i]<<endl;
            Markers[i].draw(InImage,Scalar(0,0,255),2);
        }
        cv::imshow("in",InImage);
        cv::waitKey(0);//wait for key to be pressed
    } catch (std::exception &ex)
    {
        cout<<"Exception :"<<ex.what()<<endl;
    }
}
