//
// Created by 蓬蒿浪人 on 2022/5/13.
//

#ifndef YOLO_OPENCV_YOLO_H
#define YOLO_OPENCV_YOLO_H
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

//using namespace cv;
using namespace std;
//using namespace dnn;


/*
 * 坐标转换原理：世界坐标系坐标*旋转矩阵+平移向量 = 相机坐标系坐标
 * 相机坐标系坐标/depth * 相机内参F = 图像像素坐标系
 */

enum my_color{ RED = 1 , BLUE = 2 };

struct Net_config
{
    float confThreshold; // class Confidence threshold
    float nmsThreshold;  // Non-maximum suppression threshold
    float scoThreshold;  //Object Confidence threshold
    string netname;
    string classpath;
};

struct detect_aim
{
    cv::Rect rect;
    int classid;
};

struct send_data_2d
{
    string aim_name;
    cv::Rect aim_rect;
    cv::Rect aim_armor;
};

struct send_data_xy
{
    string aim_name;
    int x;
    int y;

};

struct armor_data
{
    int name_id;
    cv::Point armor_pos;
    int width;
    int height;
};

class yolo
{
    cv::Mat F_MAT;
    cv::Mat C_MAT;
    Eigen::Matrix3d F_EGN;
    Eigen::Matrix<double,5,1> C_EGN;
    const double armor_w = 230;//mm
    const double armor_h = 127;
    const double car_w = 1000;
    const double car_h = 1000;
    double h_dis = 2000;
public:
    cv::dnn::Net net;
    float conthreshold;
    float nmsthreshold;
    float scothreshold;
    vector<string> class_names;
    int class_nums;
    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;
    int my_color;
//    vector<detect_aim> aim;

    yolo(Net_config &net_congfig);

    void detect(cv::Mat &frame);
    vector<send_data_2d> get_data(vector<int> &classids, vector<float> &confidences, vector<cv::Rect> &boxes, cv::Mat &src);

    Eigen::Vector3d pnp_get_pc(const cv::Point2f *p, const double &w, const double &h);

    vector<send_data_xy> transform_xy(vector<send_data_2d> &img_aim);

    inline Eigen::Vector3d pc_to_pu(Eigen::Vector3d& pc , double& depth)//
    {
        return F_EGN * pc / depth;
    }

    inline Eigen::Vector3d pu_to_pc(Eigen::Vector3d& pu , double& depth)
    {
        return F_EGN.inverse()*(pu * depth) ;//transpose求矩阵转置,inverse求矩阵的逆
    }
};




#endif //YOLO_OPENCV_YOLO_H
