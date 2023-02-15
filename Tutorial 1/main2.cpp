#include <iostream>
#include <opencv2/opencv.hpp>
#define IMAGE_PATH "/Users/sebila/CLionProjects/VRA/Tutorial 1/data/images/"

using namespace cv;
using namespace std;

int readDisplayImage(const String &path) {
    // exo1
    string windowName = "read/display image";
    Mat frame         = imread(path, IMREAD_UNCHANGED);
    if (frame.empty()) {
        cout << "error while reading the image" << endl;
        return 1;
    }
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, frame);
    waitKey(0);
    return 0;
}

int filtering2D(const String &path) {
    // exo2
    Mat src = imread(path, IMREAD_UNCHANGED);
    if (src.empty()) {
        cout << "error loading image..." << endl;
    }
    Mat dst;
    Mat kernel = Mat::ones(3, 3, CV_32F) / (9.);
    filter2D(src, dst, -1, kernel);
    imshow("source image", src);
    imshow("filtered image", dst);
    waitKey(0);
    return 0;
}

int smoothing(const String &path) {
    // exo3
    Mat src = imread(path, IMREAD_UNCHANGED);
    if (src.empty()) {
        cout << "error while loading image" << endl;
        return 1;
    }
    Mat dst1, dst2, dst3;
    GaussianBlur(src, dst1, Size(5, 5), 0);
    medianBlur(src, dst2, 5);
    bilateralFilter(src, dst3, 15, 30, 7.5);

    imshow("source image", src);
    imshow("gaussian blur", dst1);
    imshow("median blur", dst2);
    imshow("bilateral filter", dst3);
    waitKey(0);
    return 0;
}

int morphology(const String &path) {
    // exo4
    Mat src, erosionDst, dilatationDst;
    src = imread(path, IMREAD_UNCHANGED);
    if (src.empty()) {
        cout << "error while loading image" << endl;
        return 1;
    }
    Mat structuringElement = getStructuringElement(MORPH_RECT, Size(7, 7));

    dilate(src, dilatationDst, structuringElement);
    erode(src, erosionDst, structuringElement);

    imshow("source image", src);
    imshow("dilatation", dilatationDst);
    imshow("erosion", erosionDst);
    waitKey(0);
    return 0;
}

int thresholding(const String &path) {
    // exo5
    Mat src, gray, dst, dstInv;
    src = imread(path, IMREAD_UNCHANGED);
    if (src.empty()) {
        cout << "error while loading image" << endl;
    }

    cvtColor(src, gray, COLOR_BGR2GRAY);
    threshold(gray, dst, 100, 255, THRESH_BINARY);
    threshold(gray, dstInv, 100, 255, THRESH_BINARY_INV);

    imshow("source image", src);
    imshow("gray image", gray);
    imshow("threshold image", dst);
    imshow("threshold inv image", dstInv);
    waitKey(0);
    return 0;
}


int edgeDetection(const String &path) {

    Mat src, gray, dst1, dst2;

    string windowName_src = "source image";
    string windowName_dst1 = "laplace";
    string windowName_dst2 = "canny";

    src = imread(path, IMREAD_UNCHANGED);
    if (src.empty()) {
        cout << "error while loading image..." << endl;
    }

    namedWindow(windowName_src, WINDOW_AUTOSIZE);
    namedWindow(windowName_dst1, WINDOW_AUTOSIZE);
    namedWindow(windowName_dst2, WINDOW_AUTOSIZE);

    cvtColor(src, gray, COLOR_BGR2GRAY);

    Laplacian(gray, dst1, CV_16S, 3);
    convertScaleAbs(dst1, dst1); // transform CV_16S -> 8U [0-255] for visualization

    Canny(gray, dst2, 20, 110);

    imshow(windowName_src, src);
    imshow(windowName_dst1, dst1);
    imshow(windowName_dst2, dst2);

    waitKey(0);
    destroyAllWindows();
}


int histogramEqualization(const String &path) {
    Mat src, gray, dst;

    string windowName_src = "source image";
    string windowName_dst = "histogram equalization";

    src = imread(path, IMREAD_UNCHANGED);
    if (src.empty()) {
        cout << "error while loading image" << endl;
    }

    namedWindow(windowName_src, WINDOW_AUTOSIZE);
    namedWindow(windowName_dst, WINDOW_AUTOSIZE);

    cvtColor(src, gray, COLOR_BGR2GRAY);

    equalizeHist(gray, dst);

    imshow(windowName_src, src);
    imshow(windowName_dst, dst);

    waitKey(0);
    destroyAllWindows();

    return 0;
}

int templateMatching() {
    Mat src, temp, dst;
    Point minLoc;

    string windowName_src = "source image";
    string windowName_temp = "template image";
    string windowName_dst = "template matching";

    namedWindow(windowName_src, WINDOW_AUTOSIZE);
    namedWindow(windowName_temp, WINDOW_AUTOSIZE);

    src = imread(IMAGE_PATH "bus.jpg", IMREAD_UNCHANGED);
    temp = imread(IMAGE_PATH "bus_template.png", IMREAD_UNCHANGED);

    if (src.empty() || temp.empty()) {
        cout << "error while loading image" << endl;
        return 1;
    }

    namedWindow(windowName_src, WINDOW_AUTOSIZE);
    namedWindow(windowName_temp, WINDOW_AUTOSIZE);
    namedWindow(windowName_dst, WINDOW_AUTOSIZE);

    matchTemplate(src, temp, dst, TM_SQDIFF_NORMED);

    minMaxLoc(dst, 0, 0, &minLoc, 0);
    dst = src.clone();

    rectangle(dst, minLoc, Point(minLoc.x + temp.cols, minLoc.y + temp.rows), CV_RGB(0, 255, 0), 5);

    imshow(windowName_src, src);
    imshow(windowName_temp, temp);
    imshow(windowName_dst, dst);

    waitKey(0);
    destroyAllWindows();
    return 0;
}

int findContour() {

    Mat src, gray, dst;
    string windowName_src = "source image";
    string windowName_dst = "contours";

    namedWindow(windowName_src, WINDOW_AUTOSIZE);
    namedWindow(windowName_dst, WINDOW_AUTOSIZE);

    src = imread(IMAGE_PATH "porsche.jpg", IMREAD_UNCHANGED);
    if (src.empty()) {
        cout << "error while loading image" << endl;
        return 1;
    }
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Canny(gray, gray, 100, 200);

    vector<vector<Point>> contours;
    findContours(gray, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    dst = Mat::zeros(gray.size(), CV_8SC3);
    RNG rng(1);
    for (int i = 0; i < contours.size(); i++) {
        Scalar color =  Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(dst, contours, i, color, 2);
    }
    imshow(windowName_src, src);
    imshow(windowName_dst, dst);

    waitKey(0);
    destroyAllWindows();

    return 0;
}


int main() {
     // readDisplayImage(IMAGE_PATH "landscape.jpg");
    // filtering2D(IMAGE_PATH "plane.jpg");
     smoothing(IMAGE_PATH "monalisa.jpg");
    // morphology(IMAGE_PATH "apple.png");
    // thresholding(IMAGE_PATH "tiger.jpg");
    // edgeDetection(IMAGE_PATH "building.jpg"); // Exercice 6 : Détection de contours
    // histogramEqualization(IMAGE_PATH "bird.jpg");
    // templateMatching();
    //findContour();
    return 0;
}