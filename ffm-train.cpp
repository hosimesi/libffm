#pragma GCC diagnostic ignored "-Wunused-result"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ffm.h"

using namespace std;
using namespace ffm;

string train_help() {
  return string(
      "usage: ffm-train [options] training_set_file [model_file]\n"
      "\n"
      "options:\n"
      "-l <lambda>: set regularization parameter (default 0.00002)\n"
      "-k <factor>: set number of latent factors (default 4)\n"
      "-t <iteration>: set number of iterations (default 15)\n"
      "-r <eta>: set learning rate (default 0.2)\n"
      "-s <nr_threads>: set number of threads (default 1)\n"
      "-p <path>: set path to the validation set\n"
      "-f <path>: set path for production model file\n"
      "-m <prefix>: set key prefix for production model\n"
      "-W <path>: set path of importance weights file for training set\n"
      "-WV <path>: set path of importance weights file for validation set\n"
      "-v <fold>: set the number of folds for cross-validation\n"
      "--quiet: quiet model (no output)\n"
      "--no-norm: disable instance-wise normalization\n"
      "--no-rand: disable random update\n"
      "<training_set_file>.bin will be generated)\n"
      "--json-meta: generate a meta file if sets json file path.\n"
      "--auto-stop: stop at the iteration that achieves the best "
      "validation loss (must be used with -p)\n"
      "--auto-stop-threshold: set the threshold count for stop at the iteration"
      " that achieves the best validation loss (must be used with "
      "--auto-stop)\n");
}

struct Option {
  Option() : param(ffm_get_default_param()), nr_folds(1), do_cv(false) {}
  string tr_path, va_path, model_path, production_model_path, key_prefix;
  string iwpath_training, iwpath_validation;
  ffm_parameter param;
  ffm_int nr_folds;
  bool do_cv;
};

string basename(string path) {
  const char *ptr = strrchr(&*path.begin(), '/');
  if (!ptr)
    ptr = path.c_str();
  else
    ptr++;
  return string(ptr);
}

Option parse_option(int argc, char **argv) {
  vector<string> args;
  for (int i = 0; i < argc; i++)
    args.push_back(string(argv[i]));

  if (argc == 1)
    throw invalid_argument(train_help());

  Option opt;

  ffm_int i = 1;
  for (; i < argc; i++) {
    if (args[i].compare("-t") == 0) {
      if (i == argc - 1)
        throw invalid_argument("need to specify number of iterations after -t");
      i++;
      opt.param.nr_iters = atoi(args[i].c_str());
      if (opt.param.nr_iters <= 0)
        throw invalid_argument(
            "number of iterations should be greater than zero");
    } else if (args[i].compare("-k") == 0) {
      if (i == argc - 1)
        throw invalid_argument("need to specify number of factors after -k");
      i++;
      opt.param.k = atoi(args[i].c_str());
      if (opt.param.k <= 0)
        throw invalid_argument("number of factors should be greater than zero");
    } else if (args[i].compare("-r") == 0) {
      if (i == argc - 1)
        throw invalid_argument("need to specify eta after -r");
      i++;
      opt.param.eta = atof(args[i].c_str());
      if (opt.param.eta <= 0)
        throw invalid_argument("learning rate should be greater than zero");
    } else if (args[i].compare("-l") == 0) {
      if (i == argc - 1)
        throw invalid_argument("need to specify lambda after -l");
      i++;
      opt.param.lambda = atof(args[i].c_str());
      if (opt.param.lambda < 0)
        throw invalid_argument(
            "regularization cost should not be smaller than zero");
    } else if (args[i].compare("-s") == 0) {
      if (i == argc - 1)
        throw invalid_argument("need to specify number of threads after -s");
      i++;
      opt.param.nr_threads = atoi(args[i].c_str());
      if (opt.param.nr_threads <= 0)
        throw invalid_argument("number of threads should be greater than zero");
    } else if (args[i].compare("-v") == 0) {
      if (i == argc - 1)
        throw invalid_argument("need to specify number of folds after -v");
      i++;
      opt.nr_folds = atoi(args[i].c_str());
      if (opt.nr_folds <= 1)
        throw invalid_argument("number of folds should be greater than one");
      opt.do_cv = true;
    } else if (args[i].compare("-p") == 0) {
      if (i == argc - 1)
        throw invalid_argument("need to specify path after -p");
      i++;
      opt.va_path = args[i];
    } else if (args[i].compare("-m") == 0) {
      if (i == argc - 1)
        throw invalid_argument("need to specify model key prefix after -m");
      i++;
      opt.key_prefix = args[i];
    } else if (args[i].compare("-f") == 0) {
      if (i == argc - 1)
        throw invalid_argument(
            "need to specify production model file path after -f");
      i++;
      opt.production_model_path = args[i];
    } else if (args[i].compare("-W") == 0) {
      if (i == argc - 1)
        throw invalid_argument("need to specify weights file path after -W");
      i++;
      opt.iwpath_training = args[i];
    } else if (args[i].compare("-WV") == 0) {
      if (i == argc - 1)
        throw invalid_argument("need to specify weights file path after -W");
      i++;
      opt.iwpath_validation = args[i];
    } else if (args[i].compare("--no-norm") == 0) {
      opt.param.normalization = false;
    } else if (args[i].compare("--quiet") == 0) {
      opt.param.quiet = true;
    } else if (args[i].compare("--no-rand") == 0) {
      opt.param.random = false;
    } else if (args[i].compare("--auto-stop") == 0) {
      opt.param.auto_stop = true;
    } else if (args[i].compare("--json-meta") == 0) {
      if (i == argc - 1)
        throw invalid_argument("need to specify weights file path after -W");
      i++;
      char *json_meta_path = new char[args[i].length() + 1];
      strcpy(json_meta_path, args[i].c_str());
      opt.param.json_meta_path = json_meta_path;
    } else if (args[i].compare("--auto-stop-threshold") == 0) {
      if (i == argc - 1)
        throw invalid_argument("need to specify weights file path after -W");
      i++;
      opt.param.auto_stop_threshold = atoi(args[i].c_str());
    } else {
      break;
    }
  }

  if (i != argc - 2 && i != argc - 1)
    throw invalid_argument("cannot parse command\n");

  opt.tr_path = args[i];
  i++;

  if (i < argc) {
    opt.model_path = string(args[i]);
  } else if (i == argc) {
    opt.model_path = basename(opt.tr_path) + ".model";
  } else {
    throw invalid_argument("cannot parse argument");
  }

  return opt;
}

int train(Option opt) {
  ffm_problem *tr = ffm_read_problem(opt.tr_path.c_str());
  if (tr == nullptr) {
    cerr << "cannot load " << opt.tr_path << endl << flush;
    return 1;
  }

  ffm_importance_weights *iw = nullptr;
  if (!opt.iwpath_training.empty()) {
    iw = ffm_read_importance_weights(opt.iwpath_training.c_str());
    if (iw == nullptr) {
      cerr << "cannot load " << opt.iwpath_training << endl << flush;
      return 1;
    }

    if (iw->l != tr->l) {
      cerr << "The length of training and weights should be equal:" << endl;
      cerr << "training file:" << tr->l << endl;
      cerr << "weights file:" << iw->l << endl << flush;
      return 1;
    }
  }

  ffm_problem *va = nullptr;
  if (!opt.va_path.empty()) {
    va = ffm_read_problem(opt.va_path.c_str());
    if (va == nullptr) {
      ffm_destroy_problem(&tr);
      cerr << "cannot load " << opt.va_path << endl << flush;
      return 1;
    }
  }

  ffm_importance_weights *iwv = nullptr;
  if (!opt.iwpath_validation.empty()) {
    if (va == nullptr) {
      cerr << "please set validation file if you set validation weights file"
           << endl
           << flush;
      return 1;
    }

    iwv = ffm_read_importance_weights(opt.iwpath_validation.c_str());
    if (iwv == nullptr) {
      cerr << "cannot load " << opt.iwpath_validation << endl << flush;
      return 1;
    }

    if (iwv->l != va->l) {
      cerr << "The length of validation and validation's weights should be "
              "equal:"
           << endl;
      cerr << "validation file:" << va->l << endl;
      cerr << "validation's weights file:" << iwv->l << endl << flush;
      return 1;
    }
  }

  int status = 0;
  if (opt.do_cv) {
    ffm_cross_validation(tr, opt.nr_folds, opt.param);
  } else {
    ffm_model *model = ffm_train_with_validation(tr, va, iw, iwv, opt.param);

    status = ffm_save_model(model, opt.model_path.c_str());

    // Production model
    if (opt.production_model_path.c_str() != nullptr)
      status = ffm_save_production_model(
          model, opt.production_model_path.c_str(), opt.key_prefix.c_str());

    ffm_destroy_model(&model);
  }

  ffm_destroy_problem(&tr);
  ffm_destroy_problem(&va);

  return status;
}

int main(int argc, char **argv) {
  Option opt;
  try {
    opt = parse_option(argc, argv);
  } catch (invalid_argument &e) {
    cout << e.what() << endl;
    return 1;
  }

  return train(opt);
}
