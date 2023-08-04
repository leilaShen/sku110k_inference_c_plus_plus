#include "TensorflowModelParser.h"
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <chrono>
#include <iostream>

//#include "gflags/gflags.h"

TensorflowModelParser::TensorflowModelParser() {}

TensorflowModelParser::~TensorflowModelParser() {
  // if(session_.get() != NULL)  session_->Close();
  // session_.reset();
  if(!session_){
    delete session_;
    session_ = nullptr;
  }
}

bool TensorflowModelParser::Get_skipped_status() { return is_skipped; }

bool TensorflowModelParser::Init(std::string pb_file_path,std::vector<std::string> output_layers,
        std::vector<std::string> input_layers,std::string gpu_visible_device,float gpu_memory_fraction) {
  output_layers_ = output_layers;
  input_layers_ = input_layers;
  model_path = pb_file_path;
  is_init_ = false;
  is_skipped = false;
  gpu_visible_device_ = gpu_visible_device;
  gpu_memory_fraction_ = gpu_memory_fraction;
  
  tensorflow::Status load_graph_status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_path, &graph_def_);
  
  if (!load_graph_status.ok()) {
    LOG(ERROR) << "Failed to load Pb files";
    return false;
  }
  
  options_.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(gpu_memory_fraction_);
//  options_.config.mutable_gpu_options()->set_allow_growth(false);
  options_.config.mutable_gpu_options()->set_allow_growth(true);
//  options_.config.mutable_gpu_options()->set_visible_device_list(gpu_visible_device_);
  options_.config.mutable_gpu_options()->set_visible_device_list("");
  options_.config.set_inter_op_parallelism_threads(1);
  options_.config.set_intra_op_parallelism_threads(1);
  // session_.reset(tensorflow::NewSession(options_));
  session_ = tensorflow::NewSession(options_);
  tensorflow::Status session_create_status = session_->Create(graph_def_);
  if (!session_create_status.ok()) {
    LOG(ERROR) << "failed to create session";
    return false;
  }
  return true;
}

bool TensorflowModelParser::Run(std::vector<tensorflow::Tensor> &inputs,std::vector<tensorflow::Tensor> &outputs) {
  if (input_layers_.size() != inputs.size()) {
    LOG(ERROR) << "Number of input tensors must be equal to:  "
               << input_layers_.size();
    return false;
  }

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_pairs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::pair<std::string, tensorflow::Tensor> input_pair(
        input_layers_[i], inputs[i]);
    inputs_pairs.push_back(input_pair);
  }

  tensorflow::Status run_status = session_->Run(inputs_pairs, {output_layers_}, {}, &outputs);

  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return false;
  }
  return true;
}


