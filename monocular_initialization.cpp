#include <iostream>
#include <cmath>
#include <numeric>

#include "gms_matcher.h"
#include "monocular_initialization.hpp"

/**
 * @brief 用ORB特征点进行特征匹配，筛选出正确的匹配对
 * 
 * @param img_1                         输入图1
 * @param img_2                         输入图2
 * @param keypoints_1                   图像1中关键点的像素坐标
 * @param keypoints_2                   图像2中关键点的像素坐标
 * @param matches                       图像1,2中特征点的匹配结果
 */
void feature_match(const Mat &img_1,
                   const Mat &img_2,
                  std::vector<KeyPoint> &keypoints_1,
                  std::vector<KeyPoint> &keypoints_2,
                  std::vector<DMatch> &matches_gms)
{

    Mat descriptors_1, descriptors_2;
    // OpenCV3 中 ORB 特征点提取代码
    Ptr<FeatureDetector> detector = ORB::create(1000);
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    // OpenCV2 中 ORB 特征点提取代码
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );

    // 检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // 根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);


    // 特征点匹配方法暴力匹配（Brute Force）：直接找距离最近的点

    BFMatcher matcher_bf(NORM_HAMMING, true); //使用汉明距离度量二进制描述子，允许交叉验证
    vector<DMatch> Matches_bf;
    matcher_bf.match(descriptors_1, descriptors_2, Matches_bf);

    // 显示暴力匹配结果
    Mat BF_img;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, Matches_bf, BF_img);
    resize(BF_img, BF_img, Size(2*img_1.cols, img_1.rows));
    putText(BF_img, "Brute Force Matches", Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
    imshow("Brute Force Matches", BF_img);


    // ---------- homework1：用GMS方法筛选暴力匹配结果中正确的匹配对 --------------//
    // ----------------  代码开始 -----------------//
    std::vector<bool> vbInliers;
    int num_inliers;
    // ...
    // ----------------  代码结束-----------------//

    cout << "# Refine Matches (after GMS):" << num_inliers  << "/" << Matches_bf.size() << endl;

    Mat imageGMS;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches_gms, imageGMS);
    resize(imageGMS, imageGMS, Size(2*img_1.cols, img_1.rows));
    putText(imageGMS, "GMS Matches", Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
    imshow("GMS Matches", imageGMS);


}


/**
 * @brief 用特征匹配对进行位姿估计
 * 
 * @param keypoints_1                   图像1中关键点的像素坐标
 * @param keypoints_2                   图像2中关键点的像素坐标
 * @param points1                       图像1转化为vector形式的特征点
 * @param points2                       图像2转化为vector形式的特征点
 * @param matches                       图像1,2中特征点的匹配结果
 * @param K                             相机内参矩阵
 * @param inlierE                       经过E筛选后的特征点是否是离群点，内点记为1，离群点记为0
 * @param R                             旋转矩阵
 * @param t                             平移向量
 */
void pose_estimation_2d2d(
    const vector<KeyPoint> &keypoints_1,
    const vector<KeyPoint> &keypoints_2,
    vector<Point2d> &points1,
    vector<Point2d> &points2,
    const std::vector<DMatch> &matches,
    const Mat K, Mat &inlierE,
    Mat &R, Mat &t) {

    // 把匹配点转换为vector<Point2d>的形式
    for (int i = 0; i < (int) matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // 计算本质矩阵
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, K, cv::RANSAC, 0.99, MAX_REPROJECT_ERROR, inlierE); 

    // 从本质矩阵中恢复旋转和平移信息.
    recoverPose(essential_matrix, points1, points2, K, R, t, inlierE);
}


/**
 * @brief 特征匹配点三角化，检查是否满足初始化条件
 * 
 * @param pts_1                         图像1转化为vector形式的特征点
 * @param pts_2                         图像2转化为vector形式的特征点
 * @param R                             旋转矩阵
 * @param t                             平移向量
 * @param K                             相机内参矩阵
 * @param points                        所有三角化点的三维坐标
 * @param inlierPts                     记录三角化点是否通过筛选，内点记为1，离群点记为0
 * @return int                          成功三角化点数目
 */
int triangulation(
    const vector<Point2d> &pts_1,
    const vector<Point2d> &pts_2,
    const Mat &R, const Mat &t, const Mat&K,
    vector<Point3d> &points3d,
    Mat & inlierPts) {

    // 相机1投影矩阵 K[I|0]
    Mat T1 (3,4,CV_64F,cv::Scalar(0));
    Mat I = Mat::eye(3,3,CV_64F);
    I.copyTo(T1.rowRange(0,3).colRange(0,3));
    Mat Prj1 = K * T1;

    // 相机2投影矩阵 K[R|t]
    cv::Mat T2(3,4,CV_64F);
    R.copyTo(T2.rowRange(0,3).colRange(0,3));
    t.copyTo(T2.rowRange(0,3).col(3));
    Mat Prj2 = K*T2;

    // 三角化
    Mat pts_4d, pts_3d;
    triangulatePoints(Prj1, Prj2, pts_1, pts_2, pts_4d);

    // 转换成非齐次坐标
    convertPointsFromHomogeneous(pts_4d.t(), pts_3d);

    // ---------- homework2：筛选内外点，参考ORB-SLAM2中单目初始化代码实现 --------------//
    // ----------------  代码开始 -----------------//
    // 根据重投影误差筛选内外点
    vector<int> inlier3dPoints;
    //...

    int sum = std::accumulate(inlier3dPoints.begin(), inlier3dPoints.end(), 0);
    printf("经过重投影误差筛选后，有效3D点数为：%d / %d \n", sum, inlier3dPoints.size());

    // 根据三角化夹角来筛选合格的三维点
    vector<double> cosinPts;
    //...
    printf("经过角度筛选后，有效3D点数为： %d / %d \n",cosinPts.size() , inlier3dPoints.size());
    // ----------------  代码结束 -----------------//

    

    if(cosinPts.size() < MIN_INIT_3DPOINT_NUM){
        return 0;
    }
    printf("通过初始化最少3D点筛选！ \n");

    // 根据所有三角化角度中位数来判断是否成功初始化
    sort(cosinPts.begin(), cosinPts.end());
    double medianAngle = cosinPts[cosinPts.size() /2];
    double medianCosinThresh = cos(MEDIAN_TRIANGLE_ANGLE * 3.1416 / 180.0f);
    if (medianAngle > medianCosinThresh){
        return 0;
    }
    printf("通过初始化最小均值角度筛选！ \n");

    points3d = pts_3d;
    return  cosinPts.size();
}

/**
 * @brief 世界坐标三维点变换到像素坐标
 * 
 * @param p                             三维点
 * @param R                             旋转矩阵
 * @param t                             平移向量
 * @param K                             相机内参矩阵
 * @return Point2d                      变换后的图像像素坐标
 */
Point2d world2pixel(const Point3d &p, const Mat &R, const Mat &t, const Mat &K){
    Mat p_cam = R * (Mat_<double>(3, 1) << p.x, p.y, p.z) + t;  // world to camera
    Point2d p_uv;
    p_uv.x = K.at<double>(0, 0) * p_cam.at<double>(0) / p_cam.at<double>(2) + K.at<double>(0, 2);
    p_uv.y = K.at<double>(1, 1) * p_cam.at<double>(1) / p_cam.at<double>(2) + K.at<double>(1, 2);
    return p_uv;
}
