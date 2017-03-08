#include "array.h"
#include "utils.h"
#include "extra_func.h"
#include "of_celiu/OpticalFlow.h"
#include <opencv2/opencv.hpp>

#define UNKNOWN_FLOW_THRESH 1e9

#ifdef _MSC_VER
#define sprintf sprintf_s
#endif

void makecolorwheel(std::vector<cv::Scalar> &colorwheel)  
{  
    int RY = 15;  
    int YG = 6;  
    int GC = 4;  
    int CB = 11;  
    int BM = 13;  
    int MR = 6;  

    int i;  

    for (i = 0; i < RY; i++) colorwheel.push_back(cv::Scalar(255,       255.*i/RY,     0));  
    for (i = 0; i < YG; i++) colorwheel.push_back(cv::Scalar(255-255.*i/YG, 255,       0));  
    for (i = 0; i < GC; i++) colorwheel.push_back(cv::Scalar(0,         255,      255.*i/GC));  
    for (i = 0; i < CB; i++) colorwheel.push_back(cv::Scalar(0,         255-255.*i/CB, 255));  
    for (i = 0; i < BM; i++) colorwheel.push_back(cv::Scalar(255.*i/BM,      0,        255));  
    for (i = 0; i < MR; i++) colorwheel.push_back(cv::Scalar(255,       0,        255-255.*i/MR));  
}  

void motionToColor(cv::Mat flow, cv::Mat &color)  
{  
    if (color.empty())  
        color.create(flow.rows, flow.cols, CV_8UC3);  
    color.setTo(0);

    static std::vector<cv::Scalar> colorwheel; //Scalar r,g,b  
    if (colorwheel.empty())  
        makecolorwheel(colorwheel);  
    // determine motion range:  
    float maxrad = -1;  

    // Find max flow to normalize fx and fy  
    for (int i= 0; i < flow.rows; ++i)  {  
        for (int j = 0; j < flow.cols; ++j) {  
            cv::Vec2f flow_at_point = flow.at<cv::Vec2f>(i, j);  
            float fx = flow_at_point[0];  
            float fy = flow_at_point[1];  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
                continue;  
            float rad = sqrt(fx * fx + fy * fy);  
            maxrad = maxrad > rad ? maxrad : rad;  
        }  
    }

    for (int i= 0; i < flow.rows; ++i) {
        for (int j = 0; j < flow.cols; ++j) {  
            cv::Vec3b &data = color.at<cv::Vec3b>(i,j);
            cv::Vec2f flow_at_point = flow.at<cv::Vec2f>(i, j);  

            float fx = flow_at_point[0] / maxrad;  
            float fy = flow_at_point[1] / maxrad;  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  {  
                data[0] = data[1] = data[2] = 0;  
                continue;  
            }  
            float rad = sqrt(fx * fx + fy * fy);  

            float angle = atan2(-fy, -fx) / CV_PI;  
            float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);  
            int k0 = (int)fk;  
            int k1 = (k0 + 1) % colorwheel.size();  
            float f = fk - k0;  
            f = 0; // uncomment to see original color wheel  

            for (int b = 0; b < 3; b++)   {  
                float col0 = colorwheel[k0][b] / 255.0;  
                float col1 = colorwheel[k1][b] / 255.0;  
                float col = (1 - f) * col0 + f * col1;  
                if (rad <= 1)  
                    col = 1 - rad * (1 - col); // increase saturation with radius  
                else  
                    col *= .75; // out of range  
                data[2 - b] = (int)(255.0 * col);  
            }  
        }  
    }  
}

static std::string fns[] = {
    "5117-8_70161.jpg",
    "5117-8_70162.jpg",
    "5117-8_70163.jpg",
    "5117-8_70164.jpg",
    "5117-8_70165.jpg",
    "5117-8_70166.jpg",
    "5117-8_70167.jpg",
    "5117-8_70168.jpg",
    "5117-8_70169.jpg",
    "5117-8_70170.jpg",
    "5117-8_70171.jpg",
    "5117-8_70172.jpg",
    "5117-8_70173.jpg",
    "5117-8_70174.jpg",
    "5117-8_70175.jpg",
    "5117-8_70176.jpg",
    "5117-8_70177.jpg",
    "5117-8_70178.jpg",
    "5117-8_70179.jpg",
    "5117-8_70180.jpg",
    "5117-8_70181.jpg"
};

// frames: CV_8UC3
// labels: CV_32S
void TSP(std::vector<cv::Mat> &frames, cv::vector<cv::Mat> &labels) {
    // parameter 
    bool reestimateFlow = false;

    int n = frames.size();
    std::vector<DImage> bvxy(n);    // the optical flow
    for (int i = 1; i < n; i++) {
        DImage dim1, dim2;
        dim1.fromCvMat(frames[i-1]);
        dim2.fromCvMat(frames[i]);
        // OpticalFlow::ComputeOpticalFlow(dim1, dim2, fvxy[i-1]);
        OpticalFlow::ComputeOpticalFlow(dim2, dim1, bvxy[i-1]);
    }
    
    int frame_it = 0;
    IMG_STRUCT *img_struct_ptr;
    for (int f_id = 0; f_id < frames.size(); f_id++) {
        printf("-> Frame %d/%d\n", f_id+1, frames.size());
        frame_it = frame_it + 1;
        cv::Mat oim1 = frames[f_id].clone();
        
        if (frame_it == 1) {
            img_struct_ptr = new IMG_STRUCT();
            Init_IMG_STRUCT(*img_struct_ptr, oim1);
        } else {
            // load([root_flows flow_files(f-1).name]);
            // load get flow, bvx/y, fvxy.
            // compute_of(oim0, oim1, outname), note: [~, flow.bvx, flow.bvy] = compute_flow(oim1, oim0);
            // [~, flow.fvx, flow.fvy] = compute_flow(oim0, oim1);
            std::vector<cv::Mat> bvxy_split;
            cv::Mat flow;
            bvxy[f_id-1].toCvMat(flow);
            cv::split(flow, bvxy_split);
            cv::Mat vx = -1.0*bvxy_split[0], vy = -1.0*bvxy_split[1];
            // covert to column major memory storage
            vx = vx.t();
            vy = vy.t();
            IMG_STRUCT_Prop(*img_struct_ptr, oim1, vy, vx);
        }


        std::vector<double> E;
        int it = 0;
        img_struct_ptr->alive_dead_changed = true;
        /*IMG.SxySyy = []; IMG.Sxy = []; IMG.Syy = [];*/
        bool converged = false;
        while (!converged && it < 5 && frame_it == 1) {
            it ++;
            double old_k = img_struct_ptr->K;
            img_struct_ptr->SP_changed.setTo(1);
            double newE;
            // [IMG.K, IMG.label, IMG.SP, IMG.SP_changed, IMG.max_UID, IMG.alive_dead_changed, IMG.Sxy,IMG.Syy,IMG.SxySyy, newE] = split_move(IMG,1);
            split_move(*img_struct_ptr, 1, &newE, &img_struct_ptr->K, &img_struct_ptr->label, &img_struct_ptr->SP, &img_struct_ptr->SP_changed, &img_struct_ptr->max_uid, &img_struct_ptr->alive_dead_changed, &img_struct_ptr->Sxy, &img_struct_ptr->Syy, &img_struct_ptr->SxySyy);
            E.push_back(newE);
            converged = (img_struct_ptr->K - old_k < 2);
            // display, skipped
        }

        it = 0;
        converged = false;
        if (frame_it > 1) {
            img_struct_ptr->SP_changed.setTo(1);
            /*[IMG.K, IMG.label, IMG.SP, ~, IMG.max_UID, ~, ~, ~] = merge_move(IMG,1);
            [IMG.K, IMG.label, IMG.SP, ~, IMG.max_UID, ~, ~, ~] = split_move(IMG,10);
            [IMG.K, IMG.label, IMG.SP, ~, IMG.max_UID, ~, ~, ~] = switch_move(IMG,1);
            [IMG.K, IMG.label, IMG.SP, ~, IMG.max_UID, ~, ~, ~] = localonly_move(IMG,1000);*/
            merge_move(*img_struct_ptr, 1, NULL, &img_struct_ptr->K, &img_struct_ptr->label, &img_struct_ptr->SP, NULL, &img_struct_ptr->max_uid, NULL, NULL, NULL, NULL);
            split_move(*img_struct_ptr, 10, NULL, &img_struct_ptr->K, &img_struct_ptr->label, &img_struct_ptr->SP, NULL, &img_struct_ptr->max_uid, NULL, NULL, NULL, NULL);
            switch_move(*img_struct_ptr,1, NULL, &img_struct_ptr->K, &img_struct_ptr->label, &img_struct_ptr->SP, NULL, &img_struct_ptr->max_uid, NULL, NULL, NULL, NULL);
            localonly_move(*img_struct_ptr,1000, NULL, &img_struct_ptr->K, &img_struct_ptr->label, &img_struct_ptr->SP, NULL, &img_struct_ptr->max_uid, NULL, NULL, NULL, NULL);
        }
        img_struct_ptr->SP_changed.setTo(1);
        img_struct_ptr->alive_dead_changed = true;

        while (~converged && it < 20) {
            it ++;
            // times = zeros(1,5); for tictoc
            double newE = 0;
            cv::Mat SP_changed0, SP_changed1, SP_changed2, SP_changed3, SP_changed4;
            if (!reestimateFlow) {
                // tic;[IMG.K, IMG.label, IMG.SP, SP_changed1, IMG.max_UID, IMG.alive_dead_changed, IMG.Sxy,IMG.Syy,IMG.SxySyy,newE] = localonly_move(IMG,1500);times(2)=toc;
                localonly_move(*img_struct_ptr, 1500, &newE, &img_struct_ptr->K, &img_struct_ptr->label, &img_struct_ptr->SP, &SP_changed1, &img_struct_ptr->max_uid, &img_struct_ptr->alive_dead_changed, &img_struct_ptr->Sxy, &img_struct_ptr->Syy, &img_struct_ptr->SxySyy);
                SP_changed0 = SP_changed1.clone();
            } else {
                // tic;[IMG.K, IMG.label, IMG.SP, SP_changed0, IMG.max_UID, IMG.alive_dead_changed, IMG.Sxy,IMG.Syy,IMG.SxySyy,newE] = local_move(IMG,1000);times(1)=toc;
                // tic;[IMG.K, IMG.label, IMG.SP, SP_changed1, IMG.max_UID, IMG.alive_dead_changed, IMG.Sxy,IMG.Syy,IMG.SxySyy,newE] = localonly_move(IMG,500);times(2)=toc;
                local_move(*img_struct_ptr, 1000, &newE, &img_struct_ptr->K, &img_struct_ptr->label, &img_struct_ptr->SP, &SP_changed0, &img_struct_ptr->max_uid, &img_struct_ptr->alive_dead_changed, &img_struct_ptr->Sxy, &img_struct_ptr->Syy, &img_struct_ptr->SxySyy);
                localonly_move(*img_struct_ptr, 500, &newE, &img_struct_ptr->K, &img_struct_ptr->label, &img_struct_ptr->SP, &SP_changed1, &img_struct_ptr->max_uid, &img_struct_ptr->alive_dead_changed, &img_struct_ptr->Sxy, &img_struct_ptr->Syy, &img_struct_ptr->SxySyy);
            }
            if (frame_it>1 && it<5) {
                // tic;[IMG.K, IMG.label, IMG.SP, SP_changed2, IMG.max_UID, IMG.alive_dead_changed, IMG.Sxy,IMG.Syy,IMG.SxySyy,newE] = merge_move(IMG,1);times(3)=toc;
                // tic;[IMG.K, IMG.label, IMG.SP, SP_changed3, IMG.max_UID, IMG.alive_dead_changed, IMG.Sxy,IMG.Syy,IMG.SxySyy,newE] = split_move(IMG,1);times(4)=toc;
                // tic;[IMG.K, IMG.label, IMG.SP, SP_changed4, IMG.max_UID, IMG.alive_dead_changed, IMG.Sxy,IMG.Syy,IMG.SxySyy,newE] = switch_move(IMG,1);times(5)=toc;
                merge_move(*img_struct_ptr, 1, &newE, &img_struct_ptr->K, &img_struct_ptr->label, &img_struct_ptr->SP, &SP_changed2, &img_struct_ptr->max_uid, &img_struct_ptr->alive_dead_changed, &img_struct_ptr->Sxy, &img_struct_ptr->Syy, &img_struct_ptr->SxySyy);
                split_move(*img_struct_ptr, 1, &newE, &img_struct_ptr->K, &img_struct_ptr->label, &img_struct_ptr->SP, &SP_changed3, &img_struct_ptr->max_uid, &img_struct_ptr->alive_dead_changed, &img_struct_ptr->Sxy, &img_struct_ptr->Syy, &img_struct_ptr->SxySyy);
                switch_move(*img_struct_ptr, 1, &newE, &img_struct_ptr->K, &img_struct_ptr->label, &img_struct_ptr->SP, &SP_changed4, &img_struct_ptr->max_uid, &img_struct_ptr->alive_dead_changed, &img_struct_ptr->Sxy, &img_struct_ptr->Syy, &img_struct_ptr->SxySyy);
                // IMG.SP_changed = SP_changed0 | SP_changed1 | SP_changed2 | SP_changed3 | SP_changed4;
                img_struct_ptr->SP_changed = SP_changed0 | SP_changed1 | SP_changed2 | SP_changed3 | SP_changed4;
            } else {
                // IMG.SP_changed = SP_changed0 | SP_changed1;
                img_struct_ptr->SP_changed = SP_changed0 | SP_changed1;
            }

            E.push_back(newE);
            // consider
            // converged = ~any(
            //                  ~arrayfun(@(x)(isempty(x{1})), {IMG.SP(:).N}) 
            //             & 
            //                  IMG.SP_changed(1:IMG.K)
            //                  );
            bool cvg = true;
            for (int k = 0; k < img_struct_ptr->K; k++) {
                if (img_struct_ptr->SP_changed.at<int>(k)) {
                    cvg = false;
                    break;
                }
            }
            converged = !cvg;

            // skip display
        }
        // set sp's uid

        // find empty SP.UID and set to -1
        /*SP_UID = {IMG.SP(:).UID};
        mask = arrayfun(@(x)(isempty(x{1})), SP_UID);
        for m = find(mask)
            SP_UID{m} = -1;
        end*/
        // sp_labels(:,:,frame_it) = reshape([SP_UID{IMG.label(IMG.w+1:end-IMG.w,IMG.w+1:end-IMG.w) +1}], size(oim,1), size(oim,2));
        cv::Mat uid_mat = cv::Mat::zeros(oim1.rows, oim1.cols, CV_32S);
        uid_mat.setTo(-1);
        int w = img_struct_ptr->w;
        for (int i = 0; i < uid_mat.rows; i++) {
            for (int j = 0; j < uid_mat.cols; j++) {
                int label = img_struct_ptr->label.at<int>(w+j, w+i);
                if (label >= 0) uid_mat.at<int>(i,j) = img_struct_ptr->SP[label].UID;
            }
        }
        labels.push_back(uid_mat);
    }
}

int main() {
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> labels;
    int n = 21;
    for (int i = 0; i < n; i++) {
        cv::Mat im = cv::imread(std::string("../data/")+fns[i]);
        frames.push_back(im);
    }
    TSP(frames, labels);

    // save the labels
    for (int i = 0; i < labels.size(); i++) {
        cv::Mat vis = show_label(labels[i]);
        // cv::imshow("vis", vis);
        // cv::waitKey();
        char buf[256];
        sprintf(buf, "../save/label_%04d.png", i);
        cv::imwrite(buf, vis);
    }
    return 0;
}
