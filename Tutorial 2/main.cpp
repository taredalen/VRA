#include <iostream>
#include <opencv2/opencv.hpp>
#define MAX_IMAGES 1024

using namespace std;
using namespace cv;

VideoCapture capture;
Mat queryFrame, outputFrame, corres_img;
string windowName = "Makerless Tracking";

vector<Mat> trainImages;
vector<Mat> desc_train(MAX_IMAGES);
Mat desc_query;
vector<vector<KeyPoint>> kp_train(MAX_IMAGES);
vector<KeyPoint> kp_query(MAX_IMAGES);
vector<DMatch> inlierMatches;
Ptr<ORB> orb;
Ptr<DescriptorMatcher> descriptormatcher;
vector<KeyPoint> kp_query_inlier, kp_train_inlier;

void loadImages() {
    vector<string> files;
    glob("../data/markers", files);
    for (const auto &file : files) {
        Mat img = imread(file);
        if (img.empty()) {
            cerr << file << "is not valid ..." << endl;
            continue;
        }
        trainImages.push_back(img);
        cout << file << "loaded." << endl;
    }
}

int orbTracker(int indexTrainImage) {
    vector<vector<DMatch>> matches;
    descriptormatcher->knnMatch(desc_query, desc_train.at(indexTrainImage),
                                matches, 2);

    vector<KeyPoint> kp_query_good_match, kp_train_good_match;
    for (auto &match : matches) {
        if (match[0].distance < 0.7 * match[1].distance) {
            kp_query_good_match.push_back(kp_query[match[0].queryIdx]);
            kp_train_good_match.push_back(
                    kp_train.at(indexTrainImage)[match[0].trainIdx]);
        }
    }
    int num_good_matches = kp_query_good_match.size();

    if (num_good_matches < 4)
        return 0;

    vector<Point2f> pts_query_good_matches, pts_train_good_matches;
    for (int i = 0; i < num_good_matches; i++) {
        pts_query_good_matches.push_back(kp_query_good_match[i].pt);
        pts_train_good_matches.push_back(kp_query_good_match[i].pt);
    }
    Mat inlierMask    = Mat(1, kp_query_good_match.size(), 8U);
    Mat orbHomography = findHomography(
            pts_query_good_matches, pts_train_good_matches, RANSAC, 2, inlierMask);

    int num_inlier_matches = 0;
    for (int i = 0, j = 0; i < num_good_matches; i++) {
        if (inlierMask.at<uchar>(i)) {
            kp_query_inlier.push_back(kp_query_good_match[i]);
            kp_train_inlier.push_back(kp_train_good_match[i]);
            inlierMatches.emplace_back(j, j, 0);
            j++;
            num_inlier_matches++;
        }
    }
    return num_good_matches;
}

int main() {
    loadImages();
    if (trainImages.empty()) {
        cerr << "No image !" << endl;
        return 1;
    }
    cout << "\n total image collected : " << trainImages.size() << endl;

    orb               = ORB::create();
    descriptormatcher = DescriptorMatcher::create("BruteForce-Hamming");

    for (int i = 0; i < trainImages.size(); i++) {
        orb->detectAndCompute(trainImages[i], noArray(), kp_train[i],
                              desc_train[i]);
        cout << "Detection and description done " << endl;
    }

    capture.open("../data/videoTuto3.MOV");
    namedWindow(windowName, WINDOW_AUTOSIZE);
    while (true) {
        capture >> queryFrame;
        if (queryFrame.empty())
            break;

        resize(queryFrame, queryFrame, Size(640, 480));
        outputFrame = queryFrame.clone();
        orb->detectAndCompute(queryFrame, noArray(), kp_query, desc_query);

        for (int indexTrainImages = 0; indexTrainImages < trainImages.size();
             indexTrainImages++) {
            int numberOfMatches = orbTracker(indexTrainImages);
            if (numberOfMatches > 10) {
                Mat ClosestImage;
                if (trainImages[indexTrainImages].rows >
                    trainImages[indexTrainImages].cols) {
                    resize(trainImages[indexTrainImages], ClosestImage, Size(100, 150));
                    rectangle(ClosestImage, Point(0, 0),
                              Point(ClosestImage.cols, ClosestImage.rows),
                              CV_RGB(0, 255, 0), 3);
                    ClosestImage.copyTo(outputFrame(Rect(0, 0, 100, 150)));

                } else {
                    resize(trainImages[indexTrainImages], ClosestImage, Size(150, 100));
                    rectangle(ClosestImage, Point(0, 0),
                              Point(ClosestImage.cols, ClosestImage.rows),
                              CV_RGB(0, 255, 0), 3);
                    ClosestImage.copyTo(outputFrame(Rect(0, 0, 150, 100)));
                }
                drawMatches(trainImages[indexTrainImages], kp_train_inlier, queryFrame,
                            kp_query_inlier, inlierMatches, corres_img);
                resize(corres_img, corres_img, Size(), 0.5, 0.5);
                cout << "numberOfMatches : " << numberOfMatches << endl;
                imshow("correspondances", corres_img);
            }
        }

        imshow(windowName, outputFrame);
        if (waitKey(40) == 27)
            break;
    }

    return 0;
}