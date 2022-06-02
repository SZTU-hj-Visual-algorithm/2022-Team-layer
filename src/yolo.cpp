//
// Created by 蓬蒿浪人 on 2022/5/13.
//
#include <fstream>
#include "yolo.h"


using namespace std;
using namespace cv;
using namespace Eigen;
yolo::yolo(Net_config &net_congfig)
{
    net = dnn::readNetFromONNX(net_congfig.netname);
    if (!net.empty())
    {
        printf("not load onnx failed!!!\n");
    }
//    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
//    net.setPreferableTarget(dnn::DNN_TARGET_CPU);



    conthreshold = net_congfig.confThreshold;
    nmsthreshold = net_congfig.nmsThreshold;
    scothreshold = net_congfig.scoThreshold;
    ifstream ifs(net_congfig.classpath);
    string line;
    while (getline(ifs, line)) this->class_names.push_back(line);
    this->class_nums = class_names.size();
    F_MAT=(Mat_<double>(3, 3) << 1564.40096, 0.000000000000, 641.93179, 0.000000000000, 1564.32777, 523.40759, 0.000000000000, 0.000000000000, 1.000000000000);
    C_MAT=(Mat_<double>(1, 5) << -0.07930, 0.21700, 0.00045, 0.00033, 0.00000);
    cv2eigen(F_MAT,F_EGN);
    cv2eigen(C_MAT,C_EGN);

}

vector<send_data_2d> yolo::detect(Mat &frame)
{
    int row = frame.rows;
    int col = frame.cols;
//    cout<<row<<endl;
//    cout<<col<<endl;
    int _max = row>col ? row : col;
    Mat src(_max,_max,CV_8UC3,Scalar(0,0,0));
//    cout<<src.size()<<endl;
//    Rect mask(Point(0,0),Point(col,row));
    frame.copyTo(src(Rect (0,0,col,row)));
//    src(mask) &= frame;
//
    Mat blob;
    dnn::blobFromImage(src,blob,1.0/255.0,Size(INPUT_WIDTH,INPUT_HEIGHT),Scalar(0,0,0),true, false);
//    cout<<blob.size()<<endl;
//    cout<<blob.cols<<endl;
//    cout<<blob.rows<<endl;
    net.setInput(blob);
    vector<Mat> outs;
    net.forward(outs);
    float ratio_h = (float)src.rows / (float)INPUT_HEIGHT;
    float ratio_w = (float)src.cols / (float)INPUT_WIDTH;
    vector<int> classids;
    vector<float> confidences;
    vector<Rect> boxes;
    int net_width = class_nums + 5;  //输出的网络宽度是类别数+5
    cout<<net_width<<endl;
    float* pdata = (float*)outs[0].data;
    for (int i=0;i<outs[0].size().width;i++)
    {
            float score = pdata[4]; ;//获取每一行的box框中含有某个物体的概率
//            cout<<score<<"\t";
//            cout<<pdata[0]<<"\t"<<pdata[1]<<"\t"<<pdata[2]<<"\t"<<pdata[3]<<"\t"<<endl;
            if (score >= scothreshold) {
                cv::Mat scores(1, class_names.size(), CV_32FC1, pdata + 5);
                Point classIdPoint;
                double max_class_socre;
                minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                max_class_socre = (float)max_class_socre;
//                cout<<max_class_socre<<endl;
                if (max_class_socre >= conthreshold) {
                    float x = pdata[0];  //x
                    float y = pdata[1];  //y
                    float w = pdata[2];  //w
                    float h = pdata[3];  //h
                    int left = (x - 0.5 * w) * ratio_w;
                    int top = (y - 0.5 * h) * ratio_h;
                    classids.push_back(classIdPoint.x);
                    confidences.push_back(max_class_socre * score);
                    boxes.emplace_back(left, top, int(w * ratio_w), int(h * ratio_h));
                }
            }
            pdata += net_width;//下一行
    }

    vector<int> indices;
    dnn::NMSBoxes(boxes,confidences,scothreshold*conthreshold,nmsthreshold,indices);

    vector<int> result_classids;
    vector<float> result_confidences;
    vector<Rect> result_boxes;
    for (int indice : indices)
    {
        result_boxes.push_back(boxes[indice]);
        result_classids.push_back(classids[indice]);
        result_confidences.push_back(confidences[indice]);

    }

    vector<send_data_2d> send = get_data(result_classids,result_confidences,result_boxes, frame);
    return send;

}

vector<send_data_2d> yolo::get_data(vector<int> &classids, vector<float> &confidences, vector<Rect> &boxes, Mat &src)
{

//    for (int i=0;i<classids.size();i++)
//    {
////        rectangle(src,boxes[i],Scalar(0,classids[i]*100,255),3);
//        detect_aim target;
//        target.rect = boxes[i];
//        target.classid = classids[i];
//        aim.push_back(target);
//    }
    vector<Rect> car_boxs;
    vector<armor_data> armor_centers;
    for(int i=0;i<classids.size();i++)
    {
        armor_data armor_c;
        if (classids[i] == 0)
        {
            car_boxs.push_back(boxes[i]);
        }
        else if (classids[i] > 2)
        {
            armor_c.armor_pos.x = boxes[i].x;
            armor_c.armor_pos.y = boxes[i].y;
            armor_c.name_id = classids[i];
            armor_c.width = boxes[i].width;
            armor_c.height = boxes[i].height;
            armor_centers.push_back(armor_c);
        }
    }
    bool find_color = false;
    vector<send_data_2d> send;
    for(int i=0;i<car_boxs.size();i++)
    {
        send_data_2d aim;

        for (int j=0;j<armor_centers.size();j++)
        {
            int x = armor_centers[j].armor_pos.x;
            int y = armor_centers[j].armor_pos.y;
            int w= armor_centers[j].width;
            int h = armor_centers[j].height;
            int x_candidate = (armor_centers[j].armor_pos.x > car_boxs[i].x) && (armor_centers[j].armor_pos.x < car_boxs[i].x+car_boxs[i].width);
            int y_candidate = (armor_centers[j].armor_pos.y > car_boxs[i].y) && (armor_centers[j].armor_pos.y < car_boxs[i].y+car_boxs[i].height);
            if (x_candidate && y_candidate)
            {
                aim.aim_name = class_names[armor_centers[j].name_id];
                aim.aim_rect = car_boxs[i];
                aim.aim_armor = Rect(armor_centers[j].armor_pos.x,armor_centers[j].armor_pos.y,armor_centers[j].width,armor_centers[j].height);
                send.push_back(aim);
                find_color = true;
                break;
            }
        }

        if (find_color)
        {
            find_color = false;
        }
        else
        {
            aim.aim_name = "ignore";
            aim.aim_rect=car_boxs[i];
            aim.aim_armor = Rect();
            send.push_back(aim);
        }
    }
    return send;
}

Eigen::Vector3d yolo::pnp_get_pc(const cv::Point2f *p, const double& w, const double& h)
{
    Point2f lu, ld, ru, rd;
    //这里的三维点最后要换成场上的标记点
    vector<cv::Point3d> ps = {
            {-w / 2.0 , -h / 2.0, 0.},
            {w / 2.0 , -h / 2.0, 0.},
            {w / 2.0 , h / 2.0, 0.},
            {-w / 2.0 , h / 2.0, 0.}
    };
    if (p[0].y < p[1].y) {
        lu = p[0];
        ld = p[1];
    }
    else {
        lu = p[1];
        ld = p[0];
    }
    if (p[2].y < p[3].y) {
        ru = p[2];
        rd = p[3];
    }
    else {
        ru = p[3];
        rd = p[2];
    }

    vector<Point2f> pu;
    pu.push_back(lu);
    pu.push_back(ru);
    pu.push_back(rd);
    pu.push_back(ld);

    Mat rvec;
    Mat tvec;
    Vector3d tv;


    solvePnP(ps, pu, F_MAT, C_MAT, rvec, tvec);

    cv2eigen(tvec,tv);
    return tv;
}

vector<send_data_xy> yolo::transform_xy(vector<send_data_2d> &img_aim) {
    vector<send_data_xy> send_Data;
//    vector<Point3d> three_point ={
//            {-w/2.0, -h/2.0, 0.0},
//            {w/2.0, -h/2.0, 0.0},
//            {w/2.0, h/2.0, 0.0},
//            {-w/2.0, h/2.0, 0.0}
//    };

    for (int i = 0; i < img_aim.size(); i++) {
        send_data_xy send_xy;
        Vector3d target;
        if (img_aim[i].aim_name != "ignore") {
            Point2f p[4];
            p[0].x = (float) img_aim[i].aim_armor.x;
            p[0].y = (float) img_aim[i].aim_armor.y;
            p[1].x = (float) img_aim[i].aim_armor.x + (float) img_aim[i].aim_armor.width;
            p[1].y = (float) img_aim[i].aim_armor.y;
            p[2].x = (float) img_aim[i].aim_armor.x + (float) img_aim[i].aim_armor.width;
            p[2].y = (float) img_aim[i].aim_armor.y + (float) img_aim[i].aim_armor.height;
            p[3].x = (float) img_aim[i].aim_armor.x;
            p[3].y = (float) img_aim[i].aim_armor.y + (float) img_aim[i].aim_armor.height;
            target = pnp_get_pc(p, armor_w, armor_h);

        } else {
            Point2f p[4];
            p[0].x = (float) img_aim[i].aim_rect.x;
            p[0].y = (float) img_aim[i].aim_rect.y;
            p[1].x = (float) img_aim[i].aim_rect.x + (float) img_aim[i].aim_rect.width;
            p[1].y = (float) img_aim[i].aim_rect.y;
            p[2].x = (float) img_aim[i].aim_rect.x + (float) img_aim[i].aim_rect.width;
            p[2].y = (float) img_aim[i].aim_rect.y + (float) img_aim[i].aim_rect.height;
            p[3].x = (float) img_aim[i].aim_rect.x;
            p[3].y = (float) img_aim[i].aim_rect.y + (float) img_aim[i].aim_rect.height;
            target = pnp_get_pc(p, car_w, car_h);
        }
        send_xy.aim_name = img_aim[i].aim_name;
        double distance = target.norm();
        double xy_dis = sqrt(fabs(distance * distance - h_dis*h_dis));
        double dis_x = target(0,0);
        double dis_y = sqrt(xy_dis*xy_dis - target(0,0)*target(0,0));
        if (my_color == RED)
        {
            send_xy.y = -5358 - (int)dis_x;
            send_xy.x = (int)dis_y - 1000;
        }
        else if (my_color = BLUE)
        {
            send_xy.y = -9642 + (int)dis_x;
            send_xy.x = 28000 - ((int)dis_y - 1000);
        }
        send_Data.push_back(send_xy);
    }
    return send_Data;
}

