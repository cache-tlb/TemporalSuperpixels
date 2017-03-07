#ifndef EXTRA_FUNC
#define EXTRA_FUNC

#include <vector>
#include <opencv2/opencv.hpp>

struct PREV_INDEX_STRUCT {
    int N, allN;
    cv::Mat all, a, c, f, b, d, u;    // CV_32S  
};

struct IMG_STRUCT {
    struct HYPER_STRUCT {
        cv::Vec2d p_Sigma, p_Delta;
        cv::Vec3d a_Sigma, a_Delta;
        cv::Vec2d p_theta;
        cv::Vec3d a_theta;
        cv::Vec2d op_Sigma;
        cv::Vec3d oa_Sigma;
    };
    struct SP_STRUCT {
        cv::Vec2d p_Delta, p_theta, p_mu, v, prev_v;
        cv::Vec3d a_theta, a_Delta, a_mu;
        int UID, N, old;
        SP_STRUCT() : p_Delta(0,0), p_theta(0,0), p_mu(0,0), v(0,0), prev_v(0,0), a_theta(0,0,0), a_Delta(0,0,0), a_mu(0,0,0), N(0), UID(-1), old(0) {}
    };
    double cov_var_a, cov_var_p, oxdim, oydim, area, area_var, w, xdim, ydim, log_alpha, log_beta, dummy_log_prob, K;
    bool alive_dead_changed;
    HYPER_STRUCT hyper;
    std::vector<SP_STRUCT> SP;
    cv::Mat boundary_mask;  // binary mask, CV_32S, mat w x h
    cv::Mat label;          // CV_32S, mat,  w x h, row major
    cv::Mat data;           // CV_64F, Mat, [w*h] x 5
    cv::Mat T4Table;        // CV_64F, vec
    int max_uid;
    cv::Mat SP_changed;     // binary, CV_32S, vec
    cv::Mat Sxy, Syy, SxySyy, obs_u, obs_v; // mat,CV_64F
    double prev_K;
    cv::Mat prev_covariance, prev_precision; // mat,CV_64F
    cv::Mat prev_app_mean, prev_pos_mean; // mat,CV_64F
    cv::Mat prev_label;     // mat, CV_32S, w x h, row major

};

class TopologyTable {
public:
    static const double T4Table[256];
    static const double T8Table[256];
};

template <typename T>
void erase_std_vector(std::vector<T> &s, int i) {
    s.erase(s.begin() + i);
    return;
}

void SP_img2data(cv::Mat &img, int w, cv::Mat &data);

void holdpad(cv::Mat &x, int w, cv::Mat &padded);

void rescale_data(cv::Mat &data, cv::Vec2d p_sigma, cv::Vec3d a_Sigma);

void random_init(int xdim, int ydim, int w, int K, cv::Mat &z, double &num_z);

void Init_IMG_STRUCT(IMG_STRUCT &img_s, cv::Mat &im);

void IMG_STRUCT_Prop(IMG_STRUCT &img_s, cv::Mat &im, cv::Mat &vx, cv::Mat &vy);

void split_move(IMG_STRUCT &img_s, int its, double *new_E, double *K, cv::Mat *label, std::vector<IMG_STRUCT::SP_STRUCT> *sp, cv::Mat *sp_changed, int *max_UID, bool *alive_dead_changed, cv::Mat *Sxy, cv::Mat *Syy, cv::Mat *SxySyy);

void local_move(IMG_STRUCT &img_s, int its, double *new_E, double *K, cv::Mat *label, std::vector<IMG_STRUCT::SP_STRUCT> *sp, cv::Mat *sp_changed, int *max_UID, bool *alive_dead_changed, cv::Mat *Sxy, cv::Mat *Syy, cv::Mat *SxySyy);

void localonly_move(IMG_STRUCT &img_s, int its, double *new_E, double *K, cv::Mat *label, std::vector<IMG_STRUCT::SP_STRUCT> *sp, cv::Mat *sp_changed, int *max_UID, bool *alive_dead_changed, cv::Mat *Sxy, cv::Mat *Syy, cv::Mat *SxySyy);

void merge_move(IMG_STRUCT &img_s, int its, double *new_E, double *K, cv::Mat *label, std::vector<IMG_STRUCT::SP_STRUCT> *sp, cv::Mat *sp_changed, int *max_UID, bool *alive_dead_changed, cv::Mat *Sxy, cv::Mat *Syy, cv::Mat *SxySyy);

void switch_move(IMG_STRUCT &img_s, int its, double *new_E, double *K, cv::Mat *label, std::vector<IMG_STRUCT::SP_STRUCT> *sp, cv::Mat *sp_changed, int *max_UID, bool *alive_dead_changed, cv::Mat *Sxy, cv::Mat *Syy, cv::Mat *SxySyy);

void get_gp_covariance(cv::Mat &z, cv::Mat &mu, double cov_var_a, double cov_var_p, double hyper_p_delta, cv::Mat &covariance, cv::Mat &precision);

void populate_indices(int K, cv::Mat &label_in, std::vector<PREV_INDEX_STRUCT> &prev_label_out);

void SP_prop_init(int K, cv::Mat &label, cv::Mat &mean_x, cv::Mat &mean_y, cv::Mat &sp_vx, cv::Mat &sp_vy, cv::Mat &boundary_mask, cv::Mat &label_out);

// bool all_zero(const cv::Mat &m);    // m is CV_32S
bool all_zero(const cv::Vec2d &v);

#endif // !EXTRA_FUNC
