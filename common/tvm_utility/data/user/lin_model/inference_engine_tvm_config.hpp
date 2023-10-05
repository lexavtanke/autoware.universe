// Copyright 2021 Arm Limited and Contributors.
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

#include "tvm_utility/pipeline.hpp"

#ifndef TVM_UTILITY_INFERENCE_ENGINE_TVM_CONFIG_HPP_  // NOLINT
#define TVM_UTILITY_INFERENCE_ENGINE_TVM_CONFIG_HPP_


namespace model_zoo
{
namespace inf_test
{
namespace engine_load
{
namespace linear_model
{

static const tvm_utility::pipeline::InferenceEngineTVMConfig config {
  {
    3,
    0,
    0
  },  // modelzoo_version

  "lin_model",  // network_name
  "llvm",  // network_backend

  "./deploy_lib.so",  //network_module_path
  "./deploy_graph.json",  // network_graph_path
  "./deploy_param.params",  // network_params_path

  kDLCPU,  // tvm_device_type
  0,  // tvm_device_id

  {
    {"a", kDLFloat, 32, 1, {2, 2}},
    {"x", kDLFloat, 32, 1, {2, 2}},
    {"b", kDLFloat, 32, 1, {2, 2}}
  },  // network_inputs

  {
    {"output", kDLFloat, 32, 1, {2, 2}}
  }  // network_outputs
};

}  // namespace linear_model
}  // namespace engine_load
}  // namespace inf_test
}  // namespace model_zoo
#endif  // TVM_UTILITY_INFERENCE_ENGINE_TVM_CONFIG_HPP_  // NOLINT
