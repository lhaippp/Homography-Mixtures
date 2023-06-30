#include <iostream>
#include "SGridTracker.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <fstream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;
using namespace std;
using namespace cv;

int main(int ac, char* av[])
{
//     Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "input path to two consecutive frames")
            ("input1", po::value<std::string>(), "input image1 path")
            ("input2", po::value<std::string>(), "input image2 path")
            ;
    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 0;
    }

    std::string img1;
    std::string img2;

    if (vm.count("input1")) {
        cout << "input image-1 path: "
             << vm["input1"].as<string>() << ".\n";
        img1 = vm["input1"].as<string>();
    }

    if(vm.count("input2")) {
        cout << "input image-2 path: "
             << vm["input2"].as<string>() << ".\n";
        img2 = vm["input2"].as<string>();
    }

    Mat source = imread(img1);
    Mat target = imread(img2);

    //用法
    GridTracker gt;
    gt.trackerInit(source);
    gt.Update(source, target);
    std::vector<cv::Point2f> sourcePts = gt.preFeas;
    std::vector<cv::Point2f> targetPts = gt.trackedFeas;

    std::fstream sourceOut;
    std::fstream targetOut;

    sourceOut.open("sourceOut.txt", std::ios::out);
    for(size_t it = 0; it < sourcePts.size(); it++)
        sourceOut << sourcePts[it].x << " " << sourcePts[it].y << std::endl;
    sourceOut.close();

    targetOut.open("targetOut.txt", std::ios::out);
    for(size_t it = 0; it < targetPts.size(); it++)
        targetOut << targetPts[it].x << " " << targetPts[it].y << std::endl;
    targetOut.close();
}
