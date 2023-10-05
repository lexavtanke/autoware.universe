// Copyright 2021-2022 Arm Limited and Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gtest/gtest.h"
#include "tvm_utility/pipeline.hpp"
#include <neg_model/inference_engine_tvm_config.hpp>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>

using model_zoo::inf_test::engine_load::neg_model::config;


namespace tvm_utility
{
namespace linear_model
{

class PreProcessorLinearModel : public tvm_utility::pipeline::PreProcessor<std::vector<float>>
{
public:
  explicit PreProcessorLinearModel(tvm_utility::pipeline::InferenceEngineTVMConfig config)
  : network_input_a_width(config.network_inputs[0].node_shape[0]),
    network_input_a_height(config.network_inputs[0].node_shape[1]),
    network_input_datatype_bytes(config.network_inputs[0].tvm_dtype_bits / 8)
  {
    // Allocate input variable
    std::vector<int64_t> shape_a{network_input_a_width, network_input_a_height};
    tvm_utility::pipeline::TVMArrayContainer a{
      shape_a,
      config.network_inputs[0].tvm_dtype_code,
      config.network_inputs[0].tvm_dtype_bits,
      config.network_inputs[0].tvm_dtype_lanes,
      config.tvm_device_type,
      config.tvm_device_id};
    
    output = a;
  }

  // The cv::Mat can't be used as an input because it throws an exception when
  // passed as a constant reference
  tvm_utility::pipeline::TVMArrayContainerVector schedule(const std::vector<float> & input)
  {
    std::cerr << "preprocessor start" << std::endl;
    float input_mat[2][2];
    input_mat[0][0] = input[0];
    input_mat[0][1] = input[1];
    input_mat[1][0] = input[2];
    input_mat[1][1] = input[3];
    std::cerr << "input mat finish" << std::endl;

    // Create cv::Mat from input array
    cv::Mat a_input = cv::Mat(2, 2, CV_32F, &input_mat);
    std::cerr << "cv finish" << std::endl;
    std::cerr << "mat size " << a_input.size() << std::endl;
    std::cerr << "mat type " << a_input.type() << std::endl;
    // std::cerr << "A TVMArrayContainer " << a. << std::endl;

    std::cerr << "copy from bytes start" << std::endl;
    TVMArrayCopyFromBytes(
      output.getArray(), a_input.data,
      network_input_a_width * network_input_a_height *
        network_input_datatype_bytes);
    // TVMArrayCopyFromBytes(
    //   output.getArray(), a_input.data,
    //   2 * 2 *
    //     4);
    std::cerr << "first copy from bytes" << std::endl;
    
    std::cerr << "preprocessor finish" << std::endl;
    return {output};
  }

private:
  int64_t network_input_a_width;
  int64_t network_input_a_height;
  int64_t network_input_datatype_bytes;
  tvm_utility::pipeline::TVMArrayContainer output;
};

class PostProcessorLinearModel : public tvm_utility::pipeline::PostProcessor<std::vector<float>>
{
public:
  explicit PostProcessorLinearModel(tvm_utility::pipeline::InferenceEngineTVMConfig config)
  : network_output_width(config.network_outputs[0].node_shape[0]),
    network_output_height(config.network_outputs[0].node_shape[1]),
    network_output_datatype_bytes(config.network_outputs[0].tvm_dtype_bits / 8)
  {
  }

  std::vector<float> schedule(const tvm_utility::pipeline::TVMArrayContainerVector & input)
  {
    std::cerr << "postprocessor start" << std::endl;
    // auto l_h = network_output_width;   // Layer height
    // auto l_w = network_output_height;  // Layer width

    // Assert data is stored row-majored in input and the dtype is float
    assert(input[0].getArray()->strides == nullptr);
    assert(input[0].getArray()->dtype.bits == sizeof(float) * 8);
    std::cerr << "assert finish" << std::endl;
    std::cerr << "infer len " << network_output_width << std::endl; 
    std::cerr << "infer len " << network_output_height << std::endl;
    std::cerr << "infer len " << network_output_width * network_output_height << std::endl;

    // Copy the inference data to CPU memory
    std::vector<float> infer(
      network_output_width * network_output_height, 0.0f);
    for (size_t i = 0; i != infer.size(); ++i){
      std::cerr << "infer " << i << " element is " << infer[i] << std::endl;
    }
    std::cerr << "copy to bytes start" << std::endl;
    
    TVMArrayCopyToBytes(
      input[0].getArray(), infer.data(),
      network_output_width * network_output_height*
        network_output_datatype_bytes);
    std::cerr << "postprocessor finish" << std::endl;
    for (size_t i = 0; i != infer.size(); ++i){
      std::cerr << "infer " << i << " element is " << infer[i] << std::endl;
    }
    return infer;
  }

private:
  int64_t network_output_width;
  int64_t network_output_height;
  int64_t network_output_datatype_bytes;
};

TEST(PipelineExamples, SimplePipeline)
{
  std::cerr << "neg model" << std::endl;
  std::cout << "start test" << std::endl;
  // // Instantiate the pipeline
  using PrePT = PreProcessorLinearModel;
  using IET = tvm_utility::pipeline::InferenceEngineTVM;
  using PostPT = PostProcessorLinearModel;

  PrePT PreP{config};
  // std::string home_dir = getenv("HOME");
  // std::string autoware_data = "/autoware_data/";
  // IET IE{config, "tvm_utility", home_dir + autoware_data};
  IET IE{config, "tvm_utility"};
  PostPT PostP{config};

  tvm_utility::pipeline::Pipeline<PrePT, IET, PostPT> pipeline(PreP, IE, PostP);

  auto version_status = IE.version_check({2, 0, 0});
  EXPECT_NE(version_status, tvm_utility::Version::Unsupported);

  // // test


  // // printf("Prepare data for pipeline\n");
  // // create 2,2  array with numbers 1,2,3,4
  // std::vector<float> input_arr {1., 2., 3., 4.};
  std::vector<float> input_arr {1., -2., 3., -4.};
  std::cerr << "print input "  << std::endl;
  std::cerr << "first element " << input_arr[0] << std::endl;
  std::cerr << "second element " << input_arr[1] << std::endl;
  std::cerr << "third element " << input_arr[2] << std::endl;
  std::cerr << "forth element " << input_arr[3] << std::endl;
  // // feed it to the pipelined
  auto output = pipeline.schedule(input_arr);
  // EXPECT_EQ(true, false);
  // // printf("Prepare expected output\n");

  // // define output vector with expected values 2, 6, 12, 20 
  // std::vector<float> expected_output{-1., -2., -3., -4.};
  std::vector<float> expected_output{-1., 2., -3., 4.};

  std::cerr << "result output "  << std::endl;
  std::cerr << "first element " << output[0] << " expected " << expected_output[0] << std::endl;
  std::cerr << "second element " << output[1] << " expected " << expected_output[1] << std::endl;
  std::cerr << "third element " << output[2] << " expected " << expected_output[2] << std::endl;
  std::cerr << "forth element " << output[3] << " expected " << expected_output[3] << std::endl;


  // // A memcpy means that the floats in expected_output have a well-defined binary value
  // // for (size_t i = 0; i < int_output.size(); i++) {
  // //   memcpy(&expected_output[i], &int_output[i], sizeof(expected_output[i]));
  // // }
  // // printf("Check size of output\n");

  // // Test: check if the generated output is equal to the reference
  EXPECT_EQ(expected_output.size(), output.size()) << "Unexpected output size";
  // // printf("Check data\n");
  // EXPECT_EQ(true, false);
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(expected_output[i], output[i], 0.0001) << "at index: " << i;
  }

}

}  // namespace yolo_v2_tiny
}  // namespace tvm_utility
