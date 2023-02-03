#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>


using namespace cv;
using namespace std;

string path = "/Users/sebila/CLionProjects/VRA/Tutorial 4/data/";

VideoCapture capture, overlayVideo;
Mat frame, trainImage;
Ptr<AKAZE> akaze;
vector<KeyPoint> kp_train, kp_query;
Mat desc_train, desc_query;
Ptr<DescriptorMatcher> matcher;

Mat akazeHomography;
vector<Point2f> objectCorners, sceneCorners;

int akezeTraker() {
    akaze->detectAndCompute(frame, noArray(), kp_query, desc_query);


    int num_good_match = 0;
    vector<KeyPoint> kp_query_good_match, kp_train_good_match;
    vector<vector<DMatch>> matches;
    matcher->knnMatch(desc_query, desc_train, matches, 2);

    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < 0.8 * matches[i][1].distance) {
            kp_query_good_match.push_back(kp_query[matches[i][0].queryIdx]);
            kp_train_good_match.push_back(kp_train[matches[i][0].trainIdx]);
            num_good_match++;

        }
    }



    if (num_good_match < 4) return 0;

    vector<Point2f> pts_query_good_match, pts_train_good_match;

    for (int i = 0; i < num_good_match; i++) {
        pts_query_good_match.push_back(kp_query_good_match[i].pt);
        pts_train_good_match.push_back(kp_train_good_match[i].pt);
    }
    akazeHomography = findHomography(pts_train_good_match, pts_query_good_match, RANSAC, 2);

    return num_good_match;
}

void overlay() {
    Mat overlayFrame, warpOverlayFrame;
    overlayVideo >> overlayFrame;

    if(overlayVideo.get(CAP_PROP_POS_FRAMES) == overlayVideo.get(CAP_PROP_FRAME_COUNT)) {
        overlayVideo.set(CAP_PROP_POS_FRAMES, 0);
    }

    resize(overlayFrame, overlayFrame, trainImage.size());

    Mat trakingHomography = findHomography(objectCorners, sceneCorners);

    warpPerspective(overlayFrame, warpOverlayFrame, trakingHomography, frame.size());
    // imshow("", warpOverlayFrame);

    vector<Point> vec_sceneCorners(sceneCorners.begin(), sceneCorners.end());
    vector<vector<Point>> poly_rect_object;
    poly_rect_object.push_back(vec_sceneCorners);
    fillPoly(frame, poly_rect_object, Scalar(0));
    frame = warpOverlayFrame + frame;
}

int main() {

    overlayVideo.open( path + "Mercedes-C-Class.mp4");

    trainImage = imread("/Users/sebila/CLionProjects/VRA/Tutorial 4/data/Mercedes-C-Class.png", IMREAD_UNCHANGED);
    if (trainImage.empty()) {
        cerr << "Image not found!" << endl;
        return 1;
    }

    objectCorners = { Point2f(0, 0), Point2f(trainImage.cols, 0), Point2f(trainImage.cols, trainImage.rows),  Point2f(0, trainImage.rows)};
    akaze = AKAZE::create();
    matcher = DescriptorMatcher::create("BruteForce-Hamming");
    akaze->detectAndCompute(trainImage, noArray(), kp_train, desc_train);

    capture.open(path + "videoTuto5.avi");
    if (!capture.isOpened()) {
        cerr << "Can not open the video!" << endl;
        return 1;
    }
    
    while(true) {
        capture >> frame;
        if (frame.empty()) break;

        if(akezeTraker() > 20) {
            perspectiveTransform(objectCorners, sceneCorners, akazeHomography);

            //for (int i = 0; i < sceneCorners.size(); i++) {
            //    circle(frame, sceneCorners[i], 3, CV_RGB(255, 0, 0));
            //}
            overlay();
        }

        imshow("2D AR  Registration", frame);
        if (waitKey(1) == 27) break;
    }
    return 0;
}
