// mod.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <filesystem>


std::list<std::string> getImages(std::string path);
cv::Mat preprocess(std::string path);

int main()
{
    cv::dnn::Net model = cv::dnn::readNetFromTensorflow("C:\\Users\\imuzychenko\\Downloads\\simple_frozen_graph.pb");
    std::list<std::string> path_list = getImages("C:\\Dev\\tess+model\\test\\img\\digital\\0");
}

std::list<std::string> getImages(std::string path) {
    //collect paths from the folder
    std::list<std::string> files;
    for (const auto& file : std::filesystem::directory_iterator(path)) {
        files.push_back(file.path().u8string());
    }
    return files;
}

cv::Mat preprocess(std::string path) {
    //read an image in grayscale
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    cv::Mat output, resizedImg;
    //resize image to 28x28 needed for model input
    cv::resize(image, resizedImg, cv::Size(28, 28));
    cv::threshold(resizedImg, resizedImg, 80, 255, cv::THRESH_BINARY);
    // cv::GaussianBlur(resizedImg, resizedImg,cv::Size(5,5), 0.0);
    cv::imshow("s", resizedImg); // for debug
    //normalizing the image from 0 to 1
    resizedImg.convertTo(output, CV_32F, 1.0 / 255, 0);
    cv::imshow("m", output);//for debug
    cv::Mat blob = cv::dnn::blobFromImage(image,1.0, cv::Size(28, 28));
    return blob;
}