#ifndef TENSORFLOW_MODEL_PARSER_H
#define TENSORFLOW_MODEL_PARSER_H

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/session.h>
#include <memory>
#include <opencv2/opencv.hpp>

class TensorflowModelParser {
 public:
  TensorflowModelParser();
  ~TensorflowModelParser();
  bool Init(std::string pb_file_path, std::vector<std::string> output_layers,std::vector<std::string> input_layers,std::string gpu_visible_device, float gpu_memory_fraction);
  bool Run(std::vector<tensorflow::Tensor> &inputs, std::vector<tensorflow::Tensor> &outputs);
  bool Get_skipped_status();

 private:
  // the path for .pb file
  std::string model_path;

  std::string gpu_visible_device_;
  float gpu_memory_fraction_;

  // whether TensorflowModelParser object is initiated or not.
  bool is_init_ = false;

  // skip inference step or not
  bool is_skipped = false;

  // TensorFlow session to run forward prediction.
  // std::unique_ptr<tensorflow::Session> session_;
  tensorflow::Session *session_;

  // TensorFlow graph definition.
  tensorflow::GraphDef graph_def_;

  // TensorFlow session options.
  tensorflow::SessionOptions options_;

  // The name list of output layers
  std::vector<std::string> output_layers_;

  // The name list of input layers
  std::vector<std::string> input_layers_;
};

#endif  // TENSORFLOW_MODEL_PARSER_H

