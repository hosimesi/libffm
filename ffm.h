#ifndef _LIBFFM_H
#define _LIBFFM_H

#ifdef __cplusplus
extern "C" {

namespace ffm {
#endif

typedef float ffm_float;
typedef double ffm_double;
typedef int ffm_int;
typedef long long ffm_long;

struct ffm_node {
  ffm_int f;
  ffm_int j;
  ffm_float v;
};

struct ffm_problem {
  ffm_int n;
  ffm_int l;
  ffm_int m;
  ffm_node *X;
  ffm_long *P;
  ffm_float *Y;
};

struct ffm_importance_weights {
  ffm_int l;
  ffm_float sum;
  ffm_float *W;
};

struct ffm_model {
  ffm_int n;
  ffm_int m;
  ffm_int k;
  ffm_float *W;
  bool normalization;
  // Note that best_iteration is '-1' when
  // restored from model's binary format.
  ffm_int best_iteration;
};

struct ffm_parameter {
  ffm_float eta;
  ffm_float lambda;
  ffm_int nr_iters;
  ffm_int k;
  ffm_int nr_threads;
  ffm_int auto_stop_threshold;
  ffm_float nds_rate;
  char *json_meta_path;
  bool quiet;
  bool normalization;
  bool random;
  bool auto_stop;
};

ffm_problem *ffm_read_problem(char const *path);

ffm_importance_weights *ffm_read_importance_weights(char const *path);

void ffm_destroy_problem(struct ffm_problem **prob);

ffm_int ffm_save_model(ffm_model *model, char const *path);

ffm_int ffm_save_production_model(ffm_model *model, char const *path,
                                  char const *key_prefix);

ffm_model *ffm_load_model(char const *path);

void ffm_destroy_model(struct ffm_model **model);

ffm_parameter ffm_get_default_param();

ffm_model *ffm_train_with_validation(struct ffm_problem *Tr,
                                     struct ffm_problem *Va,
                                     struct ffm_importance_weights *iws,
                                     struct ffm_importance_weights *iwvs,
                                     struct ffm_parameter param);

ffm_float ffm_predict(ffm_node *begin, ffm_node *end, ffm_model *model, ffm_float nds_rate);
ffm_float calibrate(ffm_float &x, ffm_float &nds_rate);

#ifdef __cplusplus
} // namespace ffm

} // extern "C"
#endif

#endif // _LIBFFM_H
