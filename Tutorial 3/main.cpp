#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;
using namespace dnn;

VideoCapture capture;
Mat frame;

string path = "/Users/sebila/CLionProjects/VRA/Tutorial 3/yolo/";

string windowName = "Deep Learning classification";
int num_frames = 0;
int slider_pos = 0, slider_old_pos = 0;

vector<string> classes;


void drawPredictions(int classId, float confidence, Rect box) {
    rectangle(frame, box, CV_RGB(0, 255, 0));
    string label = format("%s: %.2f", classes[classId].c_str(), confidence);
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 255, 0), 1, LINE_AA);
}

void postProcessing(vector<Mat> outs, Net net) {
    float confThreshold = 0.35;

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
    double nmsThreshold = 0.5;
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPredictions(classIds[idx], confidences[idx], box);
    }
}


int main() {
    string file = path + "classes_names_yolo.txt";

    ifstream ifs(file.c_str());
    if (!ifs.is_open()) {
        cerr << file << " not found" << endl;
        return 1;
    }

    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }

    string cfg_file =  path + "yolov4-tiny.cfg";
    string model_file = path + "yolov4-tiny.weights";

    Net net = readNet(model_file, cfg_file);
    namedWindow(windowName, WINDOW_AUTOSIZE);

    capture.open(path + "vidTuto42.wmv");

    if (!capture.isOpened()) {
        cerr << "Video is not opened" << endl;
        return 1;
    }

    Mat blob;

    while (true) {
        char key = waitKey(1);
        capture >> frame;
        if (frame.empty()) break;

        blobFromImage(frame, blob, 1., Size(416, 416), Scalar(), true);

        net.setInput(blob, "", 0.00392, Scalar(0,0,0));

        vector<string> outNames = net.getUnconnectedOutLayersNames();
        vector<Mat> outs;
        net.forward(outs, outNames);

        Point classIdPoint;

        Mat prob = net.forward();

        double confidence;

        minMaxLoc(prob, 0, &confidence, 0, &classIdPoint);

        int classId = classIdPoint.x;

        postProcessing(outs, net);

        // string label = format("%s: %.2f", classes[classId].c_str(), confidence);

        // rectangle(frame, Point(0, frame.rows - 30), Point(frame.cols, frame.rows), CV_RGB(0, 0, 0), FILLED);

        // putText(frame, label, Point(0, frame.rows-7), FONT_HERSHEY_SIMPLEX, 0.8,
        // CV_RGB(255,255,255), 2, LINE_AA);

        imshow(windowName, frame);

        if (key == 27) break;
    }
}