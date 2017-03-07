#include "utils.h"
#include "extra_func.h"
#include "IMG.h"
#include <cmath>
#include <algorithm>
#include <queue>

const double TopologyTable::T4Table[256] = {0, -1, 1, 1, -1, -1, 1, 1, 1, -1, 2, 2, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, 2, 1, -1, 1, 1, 1, -1, 2, 2, -1, -1, 2, 2, 2, -1, 3, 3, 2, -1, 2, 2, 1, -1, 2, 2, -1, -1, 2, 2, 1, -1, 2, 2, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, 2, -1, -1, 2, 2, 2, -1, 3, 3, 2, -1, 2, 2, 1, -1, 2, 2, -1, -1, 2, 2, 1, -1, 2, 2, 1, -1, 1, 1, 1, 1, 2, 1, -1, -1, 2, 1, 2, 2, 3, 2, 2, 2, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, 3, 2, 2, 2, 2, 1, 2, 2, 3, 2, -1, -1, 3, 2, 3, 3, 4, 3, 3, 3, 3, 2, 2, 2, 3, 2, -1, -1, 3, 2, 2, 2, 3, 2, 2, 2, 2, 1, 1, 1, 2, 1, -1, -1, 2, 1, 2, 2, 3, 2, 2, 2, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, 3, 2, 2, 2, 2, 1, 1, 1, 2, 1, -1, -1, 2, 1, 2, 2, 3, 2, 2, 2, 2, 1, 1, 1, 2, 1, -1, -1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1};
const double TopologyTable::T8Table[256] = {0, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 3, 3, 3, 4, 3, 3, 2, 3, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

void SP_img2data(cv::Mat &img, int w, cv::Mat &data) {
    int xdim = img.rows;
    int ydim = img.cols;
    int num_pix = (2*w+xdim)*(2*w+ydim);
    
    cv::Mat lab, pad_lab;
    cv::Mat new_im;
    if (img.type() == CV_8UC1 || img.type() == CV_8UC3) {
        img.convertTo(new_im, CV_32F, 1./255.);
    } else {
        new_im = img;
    }
    
    cv::cvtColor(new_im, lab, CV_BGR2Lab);
    holdpad(lab, w, pad_lab);
    
    data = cv::Mat::zeros(num_pix, 5, CV_64F);  // [x,y,l,a,b]
    for (int j = 0; j < pad_lab.cols; j++) {
        for (int i = 0; i < pad_lab.rows; i++) {
            int idx = j*pad_lab.rows + i;
            int x = i - w;
            int y = j - w;
            float l = pad_lab.at<cv::Vec3f>(i,j)[0], a = pad_lab.at<cv::Vec3f>(i,j)[1], b = pad_lab.at<cv::Vec3f>(i,j)[2];
            data.at<double>(idx, 0) = x;      // actual i
            data.at<double>(idx, 1) = y;      // actual j
            data.at<double>(idx, 2) = l*2.55;
            data.at<double>(idx, 3) = a + 128.;
            data.at<double>(idx, 4) = b + 128.;
        }
    }
}

void holdpad(cv::Mat &x, int w, cv::Mat &padded) {
//     int xstart = ceil((dimx-x.rows)/2.);
//     int xstop = xstart+x.rows;
//     int ystart = ceil((dimy-x.cols)/2);
//     int ystop = ystart+x.cols;
    cv::copyMakeBorder(x, padded, w, w, w, w, cv::BORDER_REPLICATE);
}

void rescale_data(cv::Mat &data, cv::Vec2d p_sigma, cv::Vec3d a_Sigma) {
    for (int i = 0; i < data.rows; i++) {
        data.at<double>(i,0) /= sqrt(p_sigma[0]);
        data.at<double>(i,1) /= sqrt(p_sigma[1]);
        data.at<double>(i,2) /= sqrt(a_Sigma[0]);
        data.at<double>(i,3) /= sqrt(a_Sigma[1]);
        data.at<double>(i,4) /= sqrt(a_Sigma[2]);
    }
}

void random_init(int xdim, int ydim, int w, int K, cv::Mat &z, double &num_z) {
    // xdim: rows; ydim: cols
    int rxdim = xdim - 2*w;
    int rydim = ydim - 2*w;

    int N = rxdim*rydim;
    std::vector<int> perm(N);
    for (int i = 0; i < N; i++) {
        perm[i] = i;
    }
    std::random_shuffle(perm.begin(), perm.end());
    std::random_shuffle(perm.begin(), perm.end());

    z = cv::Mat::zeros(ydim, xdim, CV_32S);
    z.setTo(-1);

    // std::vector<int> centers(K);
    std::list<int> cis;
    std::list<int> cjs;
    std::list<int> labels;
    for (int i = 0; i < K; i++) {
        int c = perm[i];
        int ci = c / rxdim, cj = c % rxdim;
        ci += w;
        cj += w;
        z.at<int>(ci, cj) = i + 1;
        cis.push_back(ci);
        cjs.push_back(cj);
        labels.push_back(i + 1);
    }
    // centers = randsample(N, K);
    // sample K center point, set the label[1,...K], and flood fill
    // i: row index, j: col index
    while (!cis.empty()) {
        int i = cis.front();
        int j = cjs.front();
        int label = labels.front();
        cis.pop_front();
        cjs.pop_front();
        labels.pop_front();

        if (j > 0 && z.at<int>(i,j-1) < 0) {
            z.at<int>(i,j-1) = label;
            cis.push_back(i);
            cjs.push_back(j-1);
            labels.push_back(label);
        }
        if (i > 0 && z.at<int>(i-1,j) < 0) {
            z.at<int>(i-1,j) = label;
            cis.push_back(i-1);
            cjs.push_back(j);
            labels.push_back(label);
        }
        if (j+1 < xdim && z.at<int>(i,j+1) < 0) {
            z.at<int>(i,j+1) = label;
            cis.push_back(i);
            cjs.push_back(j+1);
            labels.push_back(label);
        }
        if (i+1 < ydim && z.at<int>(i+1,j) < 0) {
            z.at<int>(i+1,j) = label;
            cis.push_back(i+1);
            cjs.push_back(j);
            labels.push_back(label);
        }
    }

    double max_z;
    cv::minMaxLoc(z, NULL, &max_z);
    num_z = max_z;  // supposed to be K
    return;
}

void Init_IMG_STRUCT(IMG_STRUCT &img_s, cv::Mat &im) {
    double cov_var_p = 1000;
    double cov_var_a = 100;
    double area_var = 400;
    double alpha = -15;
    double beta = -10;
    double deltap_scale = 1e-3;
    double deltaa_scale = 100;
    double K = 800;
    double Kpercent = 0.8;
    bool reestimateFlow = false;

    img_s.cov_var_a = cov_var_a;
    img_s.cov_var_p = cov_var_p;
    img_s.alive_dead_changed = true;

    img_s.oxdim = im.rows;
    img_s.oydim = im.cols;

    double N = img_s.oxdim*img_s.oydim;
    img_s.area = N/K;
    img_s.area_var = area_var;

    img_s.w = 2*round(2*sqrt(img_s.area/CV_PI));
    img_s.xdim = img_s.oxdim + 2*img_s.w;
    img_s.ydim = img_s.oydim + 2*img_s.w;

    img_s.boundary_mask = cv::Mat::zeros(img_s.ydim, img_s.xdim, CV_32S);
    img_s.boundary_mask.rowRange(img_s.w, img_s.boundary_mask.rows-img_s.w).colRange(img_s.w, img_s.boundary_mask.cols - img_s.w).setTo(1);

    img_s.log_alpha = alpha*img_s.area;
    img_s.log_beta = beta*img_s.area;
    double Sigma = sqr(img_s.area) / (-12.5123*img_s.log_alpha);

    img_s.hyper.p_Sigma = cv::Vec2d(Sigma, Sigma);
    img_s.hyper.p_Delta = cv::Vec2d(Sigma*2, Sigma*2)*deltap_scale;
    img_s.hyper.a_Sigma = cv::Vec3d(Sigma*2, Sigma, Sigma)*K/100.;
    img_s.hyper.a_Delta = cv::Vec3d(Sigma*20, Sigma*10, Sigma*10)/deltaa_scale;

    double r = sqrt(img_s.area / CV_PI);
    img_s.dummy_log_prob = (-0.5*sqr(r)/img_s.hyper.p_Sigma[0]) - log(2*CV_PI - img_s.hyper.p_Sigma[0]);

    img_s.hyper.p_theta = cv::Vec2d(0,0);
    img_s.hyper.a_theta = cv::Vec3d(0,0,0);
    img_s.hyper.op_Sigma = img_s.hyper.p_Sigma;
    img_s.hyper.oa_Sigma = img_s.hyper.a_Sigma;

    SP_img2data(im, img_s.w, img_s.data);
    rescale_data(img_s.data, img_s.hyper.op_Sigma, img_s.hyper.oa_Sigma);

    for (int i = 0; i < 2; i++) {
        img_s.hyper.p_theta[i] = img_s.hyper.p_theta[i] / sqrt(img_s.hyper.p_Sigma[i]);
        img_s.hyper.p_Delta[i] = img_s.hyper.p_Delta[i] / img_s.hyper.p_Sigma[i];
        img_s.hyper.p_Sigma[i] = 1;
    }
    for (int i = 0; i < 3; i++) {
        img_s.hyper.a_theta[i] = img_s.hyper.a_theta[i] / sqrt(img_s.hyper.a_Sigma[i]);
        img_s.hyper.a_Delta[i] = img_s.hyper.a_Delta[i] / img_s.hyper.a_Sigma[i];
        img_s.hyper.a_Sigma[i] = 1;
    }

    // label K init
    double num_z = -1;
    random_init(img_s.xdim, img_s.ydim, img_s.w, round(K*Kpercent), img_s.label, num_z);

    // process label
    // IMG.label(~IMG.boundary_mask) = 0; % this will be -1 in the end
    assert(img_s.boundary_mask.rows == img_s.label.rows && img_s.boundary_mask.cols == img_s.label.cols);
    for (int i = 0; i < img_s.boundary_mask.rows*img_s.boundary_mask.cols; i++) {
        if (!img_s.boundary_mask.at<int>(i)) {
            img_s.label.at<int>(i) = 0;
        }
    }
    double max_label;
    cv::minMaxLoc(img_s.label, NULL, &max_label);
    img_s.K = max_label;

    img_s.label = img_s.label-1;

    img_s.SP.clear();

    img_s.T4Table = cv::Mat::zeros(256,1,CV_64F);
    for (int i = 0; i < 256; i++) {
        img_s.T4Table.at<double>(i) = TopologyTable::T4Table[i];
    }

    img_s.max_uid = 0;
    img_s.SP_changed = cv::Mat::zeros(1, img_s.xdim*img_s.ydim, CV_32S);
    img_s.SP_changed.setTo(1);
}

void split_move(IMG_STRUCT &img_s, int its, double *new_E, double *K, cv::Mat *label, std::vector<IMG_STRUCT::SP_STRUCT> *sp, cv::Mat *sp_changed, int *max_UID, bool *alive_dead_changed, cv::Mat *Sxy, cv::Mat *Syy, cv::Mat *SxySyy) {
    IMG sp_img;
    sp_img.readFromStruct(&img_s);
    for (int i = 0; i < its; i++) {
        sp_img.move_IMG();
    }
    sp_img.writeToStruct(&img_s, new_E, K, label, sp, sp_changed, max_UID, alive_dead_changed, Sxy, Syy, SxySyy);
}

void local_move(IMG_STRUCT &img_s, int its, double *new_E, double *K, cv::Mat *label, std::vector<IMG_STRUCT::SP_STRUCT> *sp, cv::Mat *sp_changed, int *max_UID, bool *alive_dead_changed, cv::Mat *Sxy, cv::Mat *Syy, cv::Mat *SxySyy) {
    IMG sp_img;
    sp_img.readFromStruct(&img_s);
    bool converged = false;
    for (int i = 0; i < its; i++) {
        if (i%20 == 0 || i == its-1) {
            sp_img.Flow_QP2();
            converged = true;
            if (!sp_img.move_local_IMG())
                break;
        } else {
            if (!converged)
                converged = sp_img.move_local_IMG();
        }
    }
    sp_img.writeToStruct(&img_s, new_E, K, label, sp, sp_changed, max_UID, alive_dead_changed, Sxy, Syy, SxySyy);
}

void localonly_move(IMG_STRUCT &img_s, int its, double *new_E, double *K, cv::Mat *label, std::vector<IMG_STRUCT::SP_STRUCT> *sp, cv::Mat *sp_changed, int *max_UID, bool *alive_dead_changed, cv::Mat *Sxy, cv::Mat *Syy, cv::Mat *SxySyy) {
    IMG sp_img;
    sp_img.readFromStruct(&img_s);
    bool converged = false;
    for (int i = 0; i < its; i++) {
        if (i > 0 && (i%10 == 0 || i == its-1)) {
            //sp_img.Flow_QP2();
            converged = true;
            if (!sp_img.move_local_IMG())
                break;
        } else {
            if (!converged)
                converged = sp_img.move_local_IMG();
        }
    }
    sp_img.writeToStruct(&img_s, new_E, K, label, sp, sp_changed, max_UID, alive_dead_changed, Sxy, Syy, SxySyy);
}

void merge_move(IMG_STRUCT &img_s, int its, double *new_E, double *K, cv::Mat *label, std::vector<IMG_STRUCT::SP_STRUCT> *sp, cv::Mat *sp_changed, int *max_UID, bool *alive_dead_changed, cv::Mat *Sxy, cv::Mat *Syy, cv::Mat *SxySyy) {
    IMG sp_img;
    sp_img.readFromStruct(&img_s);
    for (int i = 0; i < its; i++)
        sp_img.move_merge_IMG();
    sp_img.writeToStruct(&img_s, new_E, K, label, sp, sp_changed, max_UID, alive_dead_changed, Sxy, Syy, SxySyy);
}

void switch_move(IMG_STRUCT &img_s, int its, double *new_E, double *K, cv::Mat *label, std::vector<IMG_STRUCT::SP_STRUCT> *sp, cv::Mat *sp_changed, int *max_UID, bool *alive_dead_changed, cv::Mat *Sxy, cv::Mat *Syy, cv::Mat *SxySyy) {
    IMG sp_img;
    sp_img.readFromStruct(&img_s);
    for (int i = 0; i < its; i++) {
        if (sp_img.move_switch_IMG())
            break;
    }
    sp_img.writeToStruct(&img_s, new_E, K, label, sp, sp_changed, max_UID, alive_dead_changed, Sxy, Syy, SxySyy);
}


void IMG_STRUCT_Prop(IMG_STRUCT &img_s, cv::Mat &im, cv::Mat &vx, cv::Mat &vy) {
    // delete the SPs that are only int he boundary
    /*for (int i = 86600; i < 86600+30; i++) {
        debug() << ((double*)vx.data)[i];
        debug() << ((double*)vy.data)[i];
    }*/
    int n = img_s.label.rows*img_s.label.cols;
    assert(n == img_s.boundary_mask.rows * img_s.boundary_mask.cols);
    assert(vx.rows == vy.rows && vx.cols == vy.cols);
    assert(vx.rows == img_s.oydim && vx.cols == img_s.oxdim);
    std::set<int> unique_vals_image, unique_vals_boundary, boundarySPs;
    for (int i = 0; i < n; i++) {
        int l = img_s.label.at<int>(i);
        if (img_s.boundary_mask.at<int>(i)) {
            unique_vals_image.insert(l);
        } else {
            unique_vals_boundary.insert(l);
        }
    }
    // std::set_difference(unique_vals_boundary.begin(), unique_vals_boundary.end(), unique_vals_image.begin(), unique_vals_image.end(), std::inserter(boundarySPs, boundarySPs.end()));   // not sure
    for (auto it = unique_vals_boundary.begin(); it != unique_vals_boundary.end(); it++) {
        if (unique_vals_image.count(*it) == 0) {
            boundarySPs.insert(*it);
        }
    }
    for (int i = 0; i < n; i++) {
        int l = img_s.label.at<int>(i);
        if (boundarySPs.count(l) > 0) {
            img_s.label.at<int>(i) = -1;
        }
    }
    // debug() << "before:" << img_s.K;
    /*cv::Mat vis = show_label(img_s.label);
    cv::imshow("label", vis);
    cv::waitKey();*/
    localonly_move(img_s, 0, NULL, &img_s.K, &img_s.label, &img_s.SP, NULL, NULL, NULL, NULL, NULL, NULL);
    // debug() << "after:" << img_s.K;
    
    // 1. image statistics
    SP_img2data(im, img_s.w, img_s.data);
    rescale_data(img_s.data, img_s.hyper.op_Sigma, img_s.hyper.oa_Sigma);

    // debug() << "before2:" << img_s.K;
    /*vis = show_label(img_s.label);
    cv::imshow("label", vis);
    cv::waitKey();*/
    std::vector<int> label_map(img_s.K + 1, 0);
    int ii = 0, totali = 0;
    cv::Mat mask = cv::Mat::zeros(img_s.label.rows, img_s.label.cols, CV_8U);
    for (int i = 0; i < n; i++) {
        if (img_s.label.at<int>(i) >= 0) mask.at<uchar>(i) = 1;
    }
    int cnt = 0;
    while (ii < img_s.K) {
        // if (isempty(IMG.SP(i).N) || IMG.SP(i).N==0)
        cnt += img_s.SP[ii].N;
        if (img_s.SP[ii].N == 0) {
            img_s.K--;
            erase_std_vector(img_s.SP, ii);
        } else {
            label_map[totali] = ii;
            ii++;
        }
        totali++;
    }
    // debug() << "after2:" << img_s.K << cnt;
    // label_map: range 0~..
    for (int i = 0; i < n; i++) {
        if (mask.at<uchar>(i)) {
            int old_label = img_s.label.at<int>(i);
            img_s.label.at<int>(i) = label_map[old_label];
        }
    }

    cv::Mat meanx, meany;
    meanx.create(img_s.SP.size(), 1, CV_64F);
    meany.create(img_s.SP.size(), 1, CV_64F);
    for (int i = 0; i < img_s.SP.size(); i++) {
        meanx.at<double>(i) = img_s.SP[i].p_mu[0];
        meany.at<double>(i) = img_s.SP[i].p_mu[1];
    }

    img_s.label.copyTo(img_s.prev_label);
    img_s.prev_K = img_s.K;
    img_s.prev_app_mean.create(img_s.prev_K, 3, CV_64F);
    img_s.prev_pos_mean.create(img_s.prev_K, 2, CV_64F);
    for (int i = 0; i < img_s.prev_K; i++) {
        img_s.prev_app_mean.at<double>(i, 0) = img_s.SP[i].a_mu[0];
        img_s.prev_app_mean.at<double>(i, 1) = img_s.SP[i].a_mu[1];
        img_s.prev_app_mean.at<double>(i, 2) = img_s.SP[i].a_mu[2];
        img_s.prev_pos_mean.at<double>(i, 0) = img_s.SP[i].p_mu[0];
        img_s.prev_pos_mean.at<double>(i, 1) = img_s.SP[i].p_mu[1];
    }

    cv::Mat mu = cv::Mat::zeros(img_s.prev_K, 5, CV_64F);
    double sqrt_oa_sig0 = sqrt(img_s.hyper.oa_Sigma[0]);
    double sqrt_oa_sig1 = sqrt(img_s.hyper.oa_Sigma[1]);
    double sqrt_oa_sig2 = sqrt(img_s.hyper.oa_Sigma[2]);
    double sqrt_op_sig0 = sqrt(img_s.hyper.op_Sigma[0]);
    double sqrt_op_sig1 = sqrt(img_s.hyper.op_Sigma[1]);
    for (int i = 0; i < img_s.prev_K; i++) {
        mu.at<double>(i, 0) = img_s.SP[i].p_mu[0] * sqrt_op_sig0;
        mu.at<double>(i, 1) = img_s.SP[i].p_mu[1] * sqrt_op_sig1;
        mu.at<double>(i, 2) = img_s.SP[i].a_mu[0] * sqrt_oa_sig0;
        mu.at<double>(i, 3) = img_s.SP[i].a_mu[1] * sqrt_oa_sig1;
        mu.at<double>(i, 4) = img_s.SP[i].a_mu[2] * sqrt_oa_sig2;
    }
    // [IMG.prev_covariance, IMG.prev_precision] = get_gp_covariance(IMG.label, mu, IMG.cov_var_a, IMG.cov_var_p, IMG.hyper.p_Delta(1));
    get_gp_covariance(img_s.label, mu, img_s.cov_var_a, img_s.cov_var_p, img_s.hyper.p_Delta[0], img_s.prev_covariance, img_s.prev_precision);


    cv::Mat vx_extend, vy_extend;
    holdpad(vx, img_s.w, vx_extend);
    holdpad(vy, img_s.w, vy_extend);

    // [IMG.prev_indices, ~] = populate_indices(double(IMG.prev_K), IMG.prev_label);
    std::vector<PREV_INDEX_STRUCT> prev_indices;
    populate_indices(img_s.K, img_s.prev_label, prev_indices);

    cv::Mat sp_vx = cv::Mat::zeros(img_s.SP.size(), 1, CV_64F);
    cv::Mat sp_vy = cv::Mat::zeros(img_s.SP.size(), 1, CV_64F);
    // cv::Mat sp_x = cv::Mat::zeros(img_s.SP.size(), 1, CV_64F);
    // cv::Mat sp_y = cv::Mat::zeros(img_s.SP.size(), 1, CV_64F);

    for (int i = 0; i < img_s.K; i++) {
        cv::Mat indices_k = prev_indices[i].all;
        int nik = indices_k.rows*indices_k.cols;
        img_s.SP[i].a_theta = img_s.SP[i].a_mu;

        double vxi = 0, vyi = 0;
        for (int j = 0; j < nik; j++) {
            int idx = indices_k.at<int>(j);
            vxi += vx_extend.at<double>(idx);
            vyi += vy_extend.at<double>(idx);
        }
        vxi /= nik;
        vyi /= nik;

        // consider
        int x = round(img_s.SP[i].p_mu[0]*sqrt(img_s.hyper.op_Sigma[0]));
        int y = round(img_s.SP[i].p_mu[1]*sqrt(img_s.hyper.op_Sigma[1]));
        int xi = std::max(0, std::min<int>(x, img_s.xdim-2*img_s.w - 1));
        int yi = std::max(0, std::min<int>(y, img_s.ydim-2*img_s.w - 1));

        img_s.SP[i].N = 0;
        img_s.SP[i].old = 1;

        img_s.SP[i].p_theta = img_s.SP[i].p_mu;

        if (all_zero(img_s.SP[i].prev_v)) {
            img_s.SP[i].prev_v = cv::Vec2d(vxi/sqrt_op_sig0,vyi/sqrt_op_sig1);
        } else {
            img_s.SP[i].prev_v = img_s.SP[i].v;
        }

        // consider
        img_s.SP[i].v = cv::Vec2d(vx.at<double>(yi*img_s.oxdim+xi)/sqrt_op_sig0, vy.at<double>(yi*img_s.oxdim+xi)/sqrt_op_sig1);

        sp_vx.at<double>(i) = vxi;
        sp_vy.at<double>(i) = vyi;

        // sp_x.at<double>(i) = vxi+x;
        // sp_y.at<double>(i) = vyi+y;
    }
    meanx = meanx*sqrt_op_sig0;
    meany = meany*sqrt_op_sig1;

    // label = SP_prop_init(IMG.K,IMG.label,meanx,meany,sp_vx,sp_vy,IMG.boundary_mask);
    cv::Mat label;
    SP_prop_init(img_s.K, img_s.label, meanx, meany, sp_vx, sp_vy, img_s.boundary_mask, label);

    img_s.label = label;
    double max_label;
    cv::minMaxLoc(label, NULL, &max_label);
    img_s.K = (int)max_label + 1;

    img_s.SP_changed.setTo(1);
}

void SP_prop_init(int K, cv::Mat &label_in, cv::Mat &mean_x, cv::Mat &mean_y, cv::Mat &sp_vx, cv::Mat &sp_vy, cv::Mat &boundary_mask, cv::Mat &label_out) {
    
    arr(int) label = getArrayInput<int>(label_in);
    int X = label_in.cols;
    int Y = label_in.rows;
    int N = X*Y;
    double N2 = N*N;
    arr(double) meanx = getArrayInput<double>(mean_x);
    arr(double) meany = getArrayInput<double>(mean_y);
    arr(double) vx = getArrayInput<double>(sp_vx);
    arr(double) vy = getArrayInput<double>(sp_vy);
    arr(int) mask = getArrayInput<int>(boundary_mask);

    label_out = cv::Mat::zeros(label_in.rows, label_in.cols, CV_32S);

    arr(int) newlabel = getArrayInput<int>(label_out);

    memset(newlabel, -1, sizeof(int)*N);
    for (int i=0; i<N; i++)
        if (mask[i])
            newlabel[i] = K;

    std::vector<linkedList<int> > possible_labels(N);

    for (int x=0; x<X; x++) for (int y=0; y<Y; y++)
    {
        int index = x + y*X;
        if (label[index]>=0) {
            int k = label[index];
            int newx = x + vx[k] + 0.5;
            int newy = y + vy[k] + 0.5;

            if (newx>=0 && newx<X && newy>=0 && newy<Y) {
                int newindex = newx + newy*X;
                possible_labels[newindex].addNodeEnd(k);
            }
        }
    }

    bool newRegion = false;
    for (int x=0; x<X; x++) for (int y=0; y<Y; y++) {
        int index = x + y*X;
        int count = possible_labels[index].getLength();
        if (count==1)
            newlabel[index] = possible_labels[index].popFirst();
        else if (count>1) {
            int closest_k = -1;
            double closest_distance = N2;
            while (!possible_labels[index].isempty()) {
                int k = possible_labels[index].popFirst();
                double distance = pow(meanx[k] + vx[k] - x,2) + pow(meany[k] + vy[k] - y,2);
                if (distance<closest_distance || closest_k<0) {
                    closest_distance = distance;
                    closest_k = k;
                }
            }
            newlabel[index] = closest_k;
        }
        else if (mask[index])
            newRegion = true;
    }
    if (newRegion)
        K++;

    // labels propagated, now enforce connectivity
    arr(bool) done = allocate_memory<bool>(N, false);
    std::vector<linkedList< linkedList<int>* > > indices(K);
    linkedList<int> explore;
    for (int x=0; x<X; x++) for (int y=0; y<Y; y++)
    {
        int index = x + y*X;
        if (!done[index] && newlabel[index]>=0) {
            done[index] = true;
            int curLabel = newlabel[index];
            explore.addNodeEnd(index);
            linkedList<int>* thisList = new linkedList<int>();
            indices[curLabel].addNodeEnd(thisList);

            while (!explore.isempty()) {
                index = explore.popFirst();
                thisList->addNodeEnd(index);
                int xi = index%X;
                int yi = index/X;
                if (xi>0 && !done[index-1] && newlabel[index-1]==curLabel) {
                    explore.addNodeEnd(index-1);
                    done[index-1] = true;
                }
                if (yi>0 && !done[index-X] && newlabel[index-X]==curLabel) {
                    explore.addNodeEnd(index-X);
                    done[index-X] = true;
                }
                if (xi<X-1 && !done[index+1] && newlabel[index+1]==curLabel) {
                    explore.addNodeEnd(index+1);
                    done[index+1] = true;
                }
                if (yi<Y-1 && !done[index+X] && newlabel[index+X]==curLabel) {
                    explore.addNodeEnd(index+X);
                    done[index+X] = true;
                }
            }
        }
        else if (!done[index] && newlabel[index]<0)
            done[index] = true;
    }

    int curK = K;
    for (int k=0; k<curK; k++) {
        if (indices[k].getLength()>1) {
            // find maximum length one first
            linkedListNode< linkedList<int>* >* theListNode = indices[k].getFirst();
            int maxLength = 0;
            int maxLengthi = -1;
            int i = 0;
            while (theListNode!=NULL) {
                int theLength = theListNode->getData()->getLength();
                if (theLength > maxLength) {
                    maxLength = theLength;
                    maxLengthi = i;
                }
                theListNode = theListNode->getNext();
                i++;
            }

            // mark all the other ones as not being done
            i = 0;
            theListNode = indices[k].getFirst();
            while (theListNode!=NULL) {
                if (i!=maxLengthi) {
                    linkedList<int>* theList = theListNode->getData();
                    linkedListNode<int>* theNode = theList->getFirst();

                    if (theList->getLength()<20) {
                        while (theNode!=NULL) {
                            int index = theNode->getData();
                            done[index] = false;
                            theNode = theNode->getNext();
                        }
                    }
                    else {
                        while (theNode!=NULL) {
                            int index = theNode->getData();
                            done[index] = true;
                            theNode = theNode->getNext();
                            newlabel[index] = K;
                        }
                        K++;
                    }
                }
                theListNode = theListNode->getNext();
                i++;
            }
        }
        if (indices[k].getLength()>0) {
            linkedListNode< linkedList<int>* >* theListNode = indices[k].getFirst();
            while (theListNode!=NULL) {
                delete theListNode->getData();
                theListNode = theListNode->getNext();
            }
        }
    }


    bool any_notdone = true;
    int count = 0;
    while (any_notdone) {
        any_notdone = false;
        // we can either just set to a neighboring K, or create new ones
        // this sets to a neighboring k
        for (int x=0; x<X; x++) for (int y=0; y<Y; y++)
        {
            int index = x + y*X;
            if (!done[index]) {
                done[index] = true;
                if (x<X-1 && done[index+1] && newlabel[index+1]>=0)
                    newlabel[index] = newlabel[index+1];
                else if (y<Y-1 && done[index+X] && newlabel[index+X]>=0)
                    newlabel[index] = newlabel[index+X];
                else if (x>0 && done[index-1] && newlabel[index-1]>=0)
                    newlabel[index] = newlabel[index-1];
                else if (y>0 && done[index-X] && newlabel[index-X]>=0)
                    newlabel[index] = newlabel[index-X];
                else
                    done[index] = false;
            }
        }
        for (int x=X-1; x>=0; x--) for (int y=Y-1; y>=0; y--)
        {
            int index = x + y*X;
            if (!done[index]) {
                done[index] = true;
                if (x<X-1 && done[index+1] && newlabel[index+1]>=0)
                    newlabel[index] = newlabel[index+1];
                else if (y<Y-1 && done[index+X] && newlabel[index+X]>=0)
                    newlabel[index] = newlabel[index+X];
                else if (x>0 && done[index-1] && newlabel[index-1]>=0)
                    newlabel[index] = newlabel[index-1];
                else if (y>0 && done[index-X] && newlabel[index-X]>=0)
                    newlabel[index] = newlabel[index-X];
                else {
                    done[index] = false;
                    any_notdone = true;
                }
            }
        }
        count++;
        if (count>10) {
            break;
            //mexErrMsgTxt("SP_prop_init exceeded tries\n");
        }
    }

    for (int i=0; i<N; i++) if (!done[i])
    {
        if (mask[i])
            printf("SP_prop_init exceeded tries\n");
        else
            newlabel[i] = -1;
    }

    deallocate_memory(done);
}

bool all_zero(const cv::Vec2d &v) {
    return v[0] == 0 && v[1] == 0;
}

void get_gp_covariance(cv::Mat &z, cv::Mat &mu, double cov_var_a, double cov_var_p, double iid_var, cv::Mat &covariance, cv::Mat &precision) {
    std::set<int> all_unique_z;
    int n = z.rows*z.cols;
    int *az = (int*)z.data;
    for (int i = 0; i < n; i++) {
        if (az[i] >= 0) {
            all_unique_z.insert(az[i]);
        }
    }
    int Nz = all_unique_z.size();
    covariance = cv::Mat::zeros(Nz, Nz, CV_64F);

    for (int zi = 0; zi < Nz; zi++) {
        for (int j = 0; j < Nz; j++) {
            double dp0 = mu.at<double>(zi, 0) - mu.at<double>(j, 0);
            double dp1 = mu.at<double>(zi, 1) - mu.at<double>(j, 1);
            double da0 = mu.at<double>(zi, 2) - mu.at<double>(j, 2);
            double da1 = mu.at<double>(zi, 3) - mu.at<double>(j, 3);
            double da2 = mu.at<double>(zi, 4) - mu.at<double>(j, 4);
            covariance.at<double>(zi, j) = exp(-(sqr(dp0) + sqr(dp1)) / (2*cov_var_p))*exp(-(sqr(da0) + sqr(da1) + sqr(da2)) / (2*cov_var_a));
        }
    }
    precision = (covariance + cv::Mat::eye(Nz, Nz, CV_64F)*iid_var);
    precision = precision.inv(cv::DECOMP_CHOLESKY);
}


inline bool is_border(int x, int y, int X, int Y, arr(int) label)
{
    int index = x + y*X;
    bool border = false;
    int cur_label = label[index];
    if (x>0) border = border || (cur_label!=label[index-1]);
    if (y>0) border = border || (cur_label!=label[index-X]);
    if (x<X-1) border = border || (cur_label!=label[index+1]);
    if (y<Y-1) border = border || (cur_label!=label[index+X]);
    return border;
}

inline bool is_border_f(int x, int y, int X, int Y, arr(int) label, int& border_label)
{
    int index = x + y*X;
    bool border = false;
    int cur_label = label[index];
    if (x>=X-1)
        border_label = -1;
    else
        border_label = label[index+1];
    return (x>=X-1) || (cur_label!=label[index+1]);
}
inline bool is_border_b(int x, int y, int X, int Y, arr(int) label, int& border_label)
{
    int index = x + y*X;
    bool border = false;
    int cur_label = label[index];
    if (x<=0)
        border_label = -1;
    else
        border_label = label[index-1];
    return (x<=0) || (cur_label!=label[index-1]);
}
inline bool is_border_d(int x, int y, int X, int Y, arr(int) label, int& border_label)
{
    int index = x + y*X;
    bool border = false;
    int cur_label = label[index];
    if (y>=Y-1)
        border_label = -1;
    else
        border_label = label[index+X];
    return (y>=Y-1) || (cur_label!=label[index+X]);
}
inline bool is_border_u(int x, int y, int X, int Y, arr(int) label, int& border_label)
{
    int index = x + y*X;
    bool border = false;
    int cur_label = label[index];
    if (y<=0)
        border_label = -1;
    else
        border_label = label[index-X];
    return (y<=0) || (cur_label!=label[index-X]);
}

inline void add_neighbor(int k, int border_label, int K, std::vector<bool> &neighbors, std::vector<int> &neighbors_N)
{
    if (border_label>=0)
    {
        if (!neighbors[k+border_label*K])
        {
            neighbors[k+border_label*K] = true;
            neighbors_N[k]++;
        }
    }
}

void populate_indices(int K, cv::Mat &label_in, std::vector<PREV_INDEX_STRUCT> &prev_label_out) {
    arr(int) label = getArrayInput<int>(label_in);
    int X = label_in.cols;
    int Y = label_in.rows;

    // outputs
    /*arr( bool ) neighbors = allocate_memory<bool>(K*K,0);
    arr( int ) neighbors_N = allocate_memory<int>(K,0);
    arr( int ) indices_N = allocate_memory<int>(K);
    arr( int ) indices_allN = allocate_memory<int>(K);
    arr( linkedList<int> ) indices_all = allocate_memory< linkedList<int> >(K);
    arr( linkedList<int> ) indices_a = allocate_memory< linkedList<int> >(K);
    arr( linkedList<int> ) indices_c = allocate_memory< linkedList<int> >(K);
    arr( linkedList<int> ) indices_f = allocate_memory< linkedList<int> >(K);
    arr( linkedList<int> ) indices_b = allocate_memory< linkedList<int> >(K);
    arr( linkedList<int> ) indices_d = allocate_memory< linkedList<int> >(K);
    arr( linkedList<int> ) indices_u = allocate_memory< linkedList<int> >(K);*/
    std::vector<bool> neighbors(K*K, 0);
    std::vector<int> neighbors_N(K, 0);
    std::vector<int> indices_N(K, 0);
    std::vector<int> indices_allN(K, 0);
    std::vector<linkedList<int> > indices_all(K);
    std::vector<linkedList<int> > indices_a(K);
    std::vector<linkedList<int> > indices_c(K);
    std::vector<linkedList<int> > indices_f(K);
    std::vector<linkedList<int> > indices_b(K);
    std::vector<linkedList<int> > indices_d(K);
    std::vector<linkedList<int> > indices_u(K);

    for (int x=0; x<X; x++) {
        for (int y=0; y<Y; y++) {
            int index = x + y*X;
            int k = label[index];
            if (k>=0) {
                int border_label = -1;

                indices_all[k].addNode(index);
                if (!is_border(x,y,X,Y,label))
                    indices_a[k].addNode(index);
                else
                    indices_c[k].addNode(index);

                if (is_border_f(x,y,X,Y,label, border_label)) {
                    indices_f[k].addNode(index);
                    add_neighbor(k, border_label, K, neighbors, neighbors_N);
                } 
                if (is_border_b(x,y,X,Y,label, border_label)) {
                    indices_b[k].addNode(index);
                    add_neighbor(k, border_label, K, neighbors, neighbors_N);
                }
                if (is_border_d(x,y,X,Y,label, border_label)) {
                    indices_d[k].addNode(index);
                    add_neighbor(k, border_label, K, neighbors, neighbors_N);
                }
                if (is_border_u(x,y,X,Y,label, border_label)) {
                    indices_u[k].addNode(index);
                    add_neighbor(k, border_label, K, neighbors, neighbors_N);
                }

                indices_N[k]++;
                indices_allN[k]++;
            }
        }
    }

    // indices
    prev_label_out.resize(K);
    for (int k = 0; k < K; k++) {
        prev_label_out[k].N = indices_N[k];
        prev_label_out[k].allN = indices_allN[k];
        writeIntVector(indices_all[k].getLength(), indices_all[k].getArray(), prev_label_out[k].all);
        writeIntVector(indices_a[k].getLength(), indices_a[k].getArray(), prev_label_out[k].a);
        writeIntVector(indices_c[k].getLength(), indices_c[k].getArray(), prev_label_out[k].c);
        writeIntVector(indices_f[k].getLength(), indices_f[k].getArray(), prev_label_out[k].f);
        writeIntVector(indices_b[k].getLength(), indices_b[k].getArray(), prev_label_out[k].b);
        writeIntVector(indices_d[k].getLength(), indices_d[k].getArray(), prev_label_out[k].d);
        writeIntVector(indices_u[k].getLength(), indices_u[k].getArray(), prev_label_out[k].u);
    }

    // neighbors
    /*
    plhs[1] = mxCreateCellMatrix(K,1);
    for (int k=0; k<K; k++)
    {
        mxSetCell(plhs[1],k, mxCreateNumericMatrix(1,neighbors_N[k], mxDOUBLE_CLASS, mxREAL));
        arr(double) neighbors_k = getArrayInput<double>(mxGetCell(plhs[1],k));
        int count = 0;
        for (int k2=0; k2<K; k2++)
            if (neighbors[k+k2*K])
                neighbors_k[count++] = k2;
    }*/

    /*deallocate_memory(neighbors);
    deallocate_memory(neighbors_N);
    deallocate_memory(indices_N);
    deallocate_memory(indices_allN);
    deallocate_memory(indices_all);
    deallocate_memory(indices_a);
    deallocate_memory(indices_c);
    deallocate_memory(indices_f);
    deallocate_memory(indices_b);
    deallocate_memory(indices_d);
    deallocate_memory(indices_u);*/
}

cv::Mat show_label(cv::Mat &label) {
    static std::vector<cv::Vec3b> colors;
    double max_label;
    cv::minMaxLoc(label, NULL, &max_label);
    while (colors.size() < (int)max_label + 1) {
        cv::Vec3b c;
        c[0] = rand() % 128 + 128;
        c[1] = rand() % 128 + 128;
        c[2] = rand() % 128 + 128;
        colors.push_back(c);
    }
    cv::Mat vis = cv::Mat::zeros(label.rows, label.cols, CV_8UC3);
    for (int i = 0; i < label.rows*label.cols; i++) {
        if (label.at<int>(i) >= 0) {
            vis.at<cv::Vec3b>(i) = colors[label.at<int>(i)];
        }
    }
    return vis;
}

int unique_label_count(cv::Mat &label) {
    std::set<int> s;
    for (int i = 0; i < label.rows*label.cols; i++) {
        s.insert(label.at<int>(i));
    }
    return s.size();
}
