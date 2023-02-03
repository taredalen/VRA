#include <GLUT/glut.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <zbar.h>

using namespace cv;
using namespace std;
using namespace zbar;

VideoCapture capture;
Mat frame;


void keyboard(unsigned char key, int x, int y)
{

}

void drag(int x, int y)
{

}

void mouse(int button, int state, int x, int y)
{

}

void init() {
    GLfloat yellow[] = { 1.0, 1.0, 0.0, 1.0 };
    GLfloat green[] = { 0.0, 1.0, 0.0, 1.0 };
    GLfloat white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    GLfloat direction[] = { 1.0, 1.0, 1.0, 0.0 };

    // specify material parameters for the lighting model
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, green);
    glMaterialfv(GL_FRONT, GL_SPECULAR, white);
    glMaterialf(GL_FRONT, GL_SHININESS, 50);

    // set light source parameters
    glLightfv(GL_LIGHT0, GL_AMBIENT, black);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, yellow);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white);
    glLightfv(GL_LIGHT0, GL_POSITION, direction);

    glEnable(GL_LIGHTING);   // enable light
    glEnable(GL_LIGHT0);     // turn LIGHT0 on
    glEnable(GL_DEPTH_TEST); // consider depth
}

void matToTexture(Mat image)
{
    GLuint texture_id;
    if (image.empty())
        cerr << "Error : couldn't read the image ..." << endl;
    else
    {
        glPushMatrix(); // push the current matrix down the stack
        glDisable(GL_LIGHTING); // disable lighting to preserve camera image light
        glDisable(GL_DEPTH_TEST); // disable depth comparisons
        glViewport(0, 0, image.cols, image.rows); // specify window coordinates rectangle
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, image.cols, 0, image.rows, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glGenTextures(1, &texture_id); // generate a texture
        // bind the texture to a texturing target (2D texture)
        glBindTexture(GL_TEXTURE_2D, texture_id);

        // set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // texture minifying
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // texture magnification

        Mat rgbImage, flipedImage;
        cvtColor(image, rgbImage, COLOR_BGR2RGB); // transform OpenCV BGR image to RGB (OpenGL format)
        flip(rgbImage, flipedImage, 0); // flip around x-axis

        // define a texture with the image image
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, flipedImage.cols, flipedImage.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, flipedImage.data);

        glEnable(GL_TEXTURE_2D); // define a texture
        // define a quadrilateral for texture display
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex3f(0, 0, 0);
        glTexCoord2f(0, 1); glVertex3f(0, flipedImage.rows, 0);
        glTexCoord2f(1, 1); glVertex3f(flipedImage.cols, flipedImage.rows, 0);
        glTexCoord2f(1, 0); glVertex3f(flipedImage.cols, 0, 0);
        glEnd();
        glDisable(GL_TEXTURE_2D);
        glEnable(GL_LIGHTING); // enable lighting again
        glEnable(GL_DEPTH_TEST); // depth comparisons and update the depth buffer
        glDeleteTextures(1, &texture_id); // delete the texture to free memory
        glPopMatrix(); // pop the current matrix stack
    }
}

vector<Point2f> qrPoints, prev_qrPoints;
Mat cameraRotationMatrix;
Vec3f cameraTranslation;

void poseEstimation() {
    vector<Point3f> p3d;
    p3d.push_back(Point3f(-50, 50, 0));
    p3d.push_back(Point3f(-50, -50, 0));
    p3d.push_back(Point3f(50, -50, 0));
    p3d.push_back(Point3f(50, 50, 0));
    Mat cameraMatrix = (Mat_<float>(3, 3) << 605.57, 0, 334.90, 0, 591.46, 238.77, 0, 0, 1);

    Vec3f rvec;

    solvePnP(p3d, qrPoints, cameraMatrix, noArray(), rvec, cameraTranslation);
    Rodrigues(rvec, cameraRotationMatrix);

}


int qrScanner(Mat gray) {
    // declare variables
    ImageScanner scanner;
    //vector<Point2f> qrPoints, prev_qrPoints;

    qrPoints.clear();

    string qrName;
    int num_qrPoints = 0;
    // convert image data to raw data
    uchar* raw = (uchar*)gray.data;
    // wrap image data
    Image image(gray.cols, gray.rows, "Y800", raw, gray.cols * gray.rows);
    // scan the image for barcodes
    int n = scanner.scan(image);
    // extract results
    for (Image::SymbolIterator symbol = image.symbol_begin();
         symbol != image.symbol_end(); ++symbol)
    {
        // get useful results
        qrName = symbol->get_data();
        num_qrPoints = symbol->get_location_size();
        for (int i = 0; i < num_qrPoints; i++)
            qrPoints.push_back(Point(symbol->get_location_x(i), symbol->get_location_y(i)));

        putText(frame, qrName, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(0,255,0), 2, LINE_AA);
        for (int i = 0; i < 4; i++) {
            line(frame, qrPoints[i], qrPoints[(i + 1) % 4], CV_RGB(0, 255, 0), 2, LINE_AA);
        }
        if (!prev_qrPoints.empty()) {
            if (cv::norm(prev_qrPoints, qrPoints, NORM_L2) < 5) {
                qrPoints = prev_qrPoints;

            }
        }
    }
    prev_qrPoints = qrPoints;

    return qrPoints.size();
}

void processImage() {
    capture >> frame;
    if (frame.empty()) return;
    waitKey(40);
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    if (qrScanner(gray) == 4) {
        poseEstimation();
    }

}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear color and depth buffer at each display call

    processImage();

    matToTexture(frame); // transform the camera frame into a GL texture

    glPushMatrix(); // push the current matrix down the stack

    glViewport(0, 0, frame.cols, frame.rows); // define the GL viewport with the same frame size
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float zNear = 1, zFar = 100;
    glFrustum(-1, 1, -1, 1, zNear, zFar); // define the projection view field
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 0, 0, 0, 0, 1, 0, 1, 0); // transform the modelview into a right-handed coordinate system
    float qrLength = norm(Mat(qrPoints[0]), Mat(qrPoints[1]), NORM_L2);
    float objectLength = 100;
    float cam_x_px = cameraTranslation[0] * qrLength / objectLength;
    float cam_y_px = cameraTranslation[1] * qrLength / objectLength;

    float zClip = (zFar - zNear) * cameraTranslation[2] / 2000;
    float xClip = -zClip * 2 * cam_x_px / frame.cols;
    float yClip = -zClip * 2 * cam_y_px / frame.rows;

    // add translations
    glTranslatef(xClip, yClip, zClip);

    // add rotations
    glRotatef(180, 0, 0, 1);
    float  rot_mat[16] = {
            cameraRotationMatrix.at<float>(0),
            cameraRotationMatrix.at<float>(3),
            cameraRotationMatrix.at<float>(6), 0,
            cameraRotationMatrix.at<float>(1),
            cameraRotationMatrix.at<float>(4),
            cameraRotationMatrix.at<float>(7), 0,
            cameraRotationMatrix.at<float>(2),
            cameraRotationMatrix.at<float>(5),
            cameraRotationMatrix.at<float>(8), 0,
            0, 0, 0, 1
    };


    glMultMatrixf(rot_mat);

    // add scale

    // load 3D model
    glutSolidTeapot(5); // draw a teapot

    glPopMatrix(); // pop the current matrix stack

    glFlush(); // execute commands
    glutPostRedisplay(); // redisplay
}

int main(int argc, char** argv)
{
    capture.open("/Users/sebila/CLionProjects/VRA/Tutorial 5/videoTuto6.mp4");
    if (!capture.isOpened())
    {
        cerr << "Couldn't open the video ..." << endl;
        return 1;
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(640, 480);
    glutCreateWindow("3D AR Overlay");
    init();
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(drag);
    glutMainLoop();

    return 0;
}