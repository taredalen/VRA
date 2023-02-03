#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace dnn;

VideoCapture capture;
Mat frame;
string windowName = "Deep Learning Image Classification";
int num_frames = 0;
int slider_pos = 0, slider_pos_old = 0;

string path = "/Users/sebila/CLionProjects/VRA/Tutorial3/";

int main() {
    string file = path + "googlenet/classes_names_googlenet.txt";
    ifstream ifs(file.c_str());
    if(!ifs.is_open()) cerr << file << " not found" << endl;

    vector<string> classes;
    string line;
    while(getline(ifs, line)) {
        classes.push_back(line);
        // cout << line << endl;
    }

    string cfg_file =  path + "googlenet/bvlc_googlenet.prototxt";
    string model_file = path + "googlenet/bvlc_googlenet.caffemodel";

    Net net = readNet(model_file, cfg_file);

    namedWindow(windowName, WINDOW_AUTOSIZE);
    capture.open(path + "vidTuto41.wmv");

    if (!capture.isOpened()) {
        cerr << "Error in loading the video!" << endl;
        return 1;
    }

    Mat blob;

    while (true) {
        char key = waitKey(1);
        capture >> frame;
        if (frame.empty()) break;

        blobFromImage(frame, blob, 1., Size(224, 224), Scalar(104, 117, 123), true); // true for permutation
        net.setInput(blob);

        Point classIdPoint;
        Mat prob = net.forward();
        double confidence;

        minMaxLoc(prob, 0, &confidence, 0, &classIdPoint);

        int classId = classIdPoint.x;

        string label = format("%s %.2f", classes[classId].c_str(), confidence);

        rectangle(frame, Point(0, frame.rows - 30), Point(frame.cols, frame.rows), CV_RGB(0, 0, 0), FILLED);
        putText(frame, label, Point(0, frame.rows - 7), FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 255, 255), 2, LINE_AA);

        imshow(windowName, frame);

        if (key == 27) break; // key esc
    }
    return 0;
}
