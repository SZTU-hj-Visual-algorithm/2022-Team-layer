#include "yolo.h"
#include "camera.h"
//using namespace cv;
int main()
{
    auto camera_warpper = new Camera;
    Net_config net_config;
    net_config.netname = "../best.onnx";
    net_config.nmsThreshold = 0.5;
    net_config.confThreshold = 0.5;
    net_config.scoThreshold = 0.5;
    net_config.classpath = "../classes.names";
    yolo yolov5(net_config);
    cv::Mat src;
//    int t, last_t=-1;
    yolov5.my_color = RED;

    camera_warpper->init();
    int count = 41;
    while(1)
    {
        camera_warpper->read_frame_rgb(src);
//        t= cv::getTickCount();
//        double time;
//        if (last_t!=-1)
//        {
//            time = (double)(t-last_t)/cv::getTickFrequency();
//            time = 1.0/time;
//            cv::putText(src,to_string(time),cv::Point(5,55),2,2,cv::Scalar(255,0,0));
//            last_t = t;
//        }
//        else
//        {
//            last_t = t;
//        }
        vector<send_data_2d> send;
        vector<send_data_xy> SEND_DATA;
        send = yolov5.detect(src);
//        int row = src.rows;
//        int col = src.cols;
//    cout<<row<<endl;
//    cout<<col<<endl;
//        int _max = row>col ? row : col;
//        cv::Mat img(_max,_max,CV_8UC3,cv::Scalar(0,0,0));
//        cout<<src.size()<<endl;
//    Rect mask(Point(0,0),Point(col,row));
//        src.copyTo(img(cv::Rect (0,0,col,row)));
//    src(mask) &= frame;
//
//        cv::Mat blob;
//        cv::dnn::blobFromImage(img,blob,1.0/255.0,cv::Size(yolov5.INPUT_WIDTH,yolov5.INPUT_HEIGHT),cv::Scalar(0,0,0),true, false);
//        yolov5.net.setInput(blob);
        SEND_DATA = yolov5.transform_xy(send);

        /*
         * send the SEND_DATA,and it includes aim_name(ignore, armor-blue-1......and so on)
         * and it also includes aim_x(the x location on the platform),aim_y(the y location on the platform)
         */
        for (int i=0;i<SEND_DATA.size();i++)
        {
            cout<<SEND_DATA[i].aim_name<<endl;
            cout<<SEND_DATA[i].x<<endl;
            cout<<SEND_DATA[i].y<<endl;
        }


        imshow("src",src);

        if (cv::waitKey(10) == 27)
        {
            camera_warpper->~Camera();
//            cv::destroyAllWindows();
            break;
            return 0;
        }
//        else if (cv::waitKey(200) == 's')
//        {
//            string path = "../armoro-" + to_string(count) + ".jpg";
//            cv::imwrite(path,src);
//            count ++;
//        }
    }
//    cout<<"hello world"<<endl;
    return 0;
}