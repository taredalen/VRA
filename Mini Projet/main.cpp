#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
// #include <conio.h>
#include <GLUT/glut.h>


using namespace cv;
using namespace std;
using namespace dnn;
using namespace chrono;

VideoCapture capture;
Mat frame;

string path = "/Users/sebila/CLionProjects/VRA/Mini Projet/data/";
string windowName = "Deep Learning Detection";


vector<string> object_interest{ "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cow", "horse", "sheep", "monitor" };


int num_frames = 0;
int slider_pos = 0, slider_old_pos = 0;
bool START = true;
bool show_all = true;

vector<string> classes;
vector<Scalar> colors = { Scalar(255, 255, 0), Scalar(0, 255, 0), cv::Scalar(0, 255, 255), Scalar(255, 0, 0) };

void drawPredictions(int classId, float confidence,Rect box) {
    Scalar_<double> color = colors[classId % colors.size()];
    string label = format("%s: %.2f", classes[classId].c_str(), confidence);
    rectangle(frame, box, color, 1);
    rectangle(frame, Point(box.x, box.y + 30), Point(box.x + box.width, box.y), color, FILLED);
    putText(frame, label, Point(box.x + 5, box.y + 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2, LINE_AA);

}

void postProcessing(vector<Mat> outs, Net net) {
    float confThreshold = 0.2;
    double nmsThreshold = 0.4;

    vector<int> classIds;
    vector<float>confidences;
    vector<Rect> boxes;

    for (int i = 0; i < outs.size(); i++) {
        Mat outBlob = Mat(outs[i].size(), outs[i].depth(), outs[i].data);

        for (int j = 0; j < outBlob.rows; j++) {
            Mat scores = outBlob.row(j).colRange(5, outBlob.cols);
            Point classIdPoint;
            double conf;
            minMaxLoc(scores, 0, &conf, 0, &classIdPoint);
            if (conf > confThreshold) {
                int centerX = outBlob.row(j).at<float>(0) * frame.cols;
                int centerY = outBlob.row(j).at<float>(1) * frame.rows;
                int width = outBlob.row(j).at<float>(2) * frame.cols;
                int height = outBlob.row(j).at<float>(3) * frame.rows;

                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back(conf);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);


    for (int i = 0; i < indices.size(); i++) {
        if (!show_all && find(object_interest.begin(), object_interest.end(), classes[classIds[indices[i]]]) == object_interest.end()) continue;
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPredictions(classIds[idx], confidences[idx], box);
    }
}

int load_classes() {
    string file = path + "classes.txt";
    ifstream ifs(file.c_str());
    if (!ifs.is_open()) {
        cerr << file << "file not found" << endl;
        return 1;
    }
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }
    return 0;
}

Net load_net() {
    string cfg_file = path + "yolov4-leaky-416.cfg";
    string model_file = path + "yolov4-leaky-416.weights";
    return readNet(model_file, cfg_file);
}

int main() {
    load_classes();
    Net net = load_net();

    capture.open(path + "sample3.mp4");
    if (!capture.isOpened()) {
        cerr << "Video is not opened" << endl;
        return 1;
    }

    namedWindow(windowName, 0);

    Mat blob;

    while (true) {
        char key = waitKey(1);
        capture >> frame;
        if (frame.empty()) break;

        blobFromImage(frame, blob, 1., Size(416, 416), Scalar(), true);
        net.setInput(blob, "", 0.00392, Scalar(0, 0, 0));

        vector<string> outNames = net.getUnconnectedOutLayersNames();
        vector<Mat> outs;
        net.forward(outs, outNames);

        Point classIdPoint;

        Mat prob = net.forward();
        double confidence;

        minMaxLoc(prob, 0, &confidence, 0, &classIdPoint);

        auto start = std::chrono::high_resolution_clock::now();

        postProcessing(outs, net);

        auto end = std::chrono::high_resolution_clock::now();
        int time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        string processing_time = format("Processing Time: %dms", time_elapsed);

        rectangle(frame, Point(0, frame.rows - 30), Point(frame.cols, frame.rows), CV_RGB(0, 0, 0), FILLED);
        putText(frame, processing_time, Point(0, frame.rows - 7), FONT_HERSHEY_SIMPLEX, 0.8,
                CV_RGB(255, 255, 255), 2, LINE_AA);

        imshow(windowName, frame);
        resizeWindow(windowName, 800, 600);


        bool no = true;

        if (key == 27) break;
        if (key != 27 && no) {
            show_all = !show_all;
            no = !no;
            cerr <<  key << endl;
            cerr <<  "key pressed!" << endl;
        }
    }
}