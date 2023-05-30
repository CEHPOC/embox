#include "../../../build/extbld/third_party/tflm/tflm/tflite-micro-main/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "^BUILD/extbld/^MOD_PATH/main/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "^BUILD/extbld/^MOD_PATH/main/tensorflow/lite/micro/micro_interpreter.h"
#include "^BUILD/extbld/^MOD_PATH/main/tensorflow/lite/schema/schema_generated.h"
#include "version.h"

#include "model_data.h"

#include "micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
	tflite::MicroErrorReporter micro_error_reporter;
	tflite::ErrorReporter* error_reporter = &micro_error_reporter;

	const tflite::Model* model = ::tflite::GetModel(model_tflite);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
  		TF_LITE_REPORT_ERROR(error_reporter,
      		"Model provided is schema version %d not equal "
     		"to supported version %d.\n",
      		model->version(), TFLITE_SCHEMA_VERSION);
	}

	tflite::MicroMutableOpResolver resolver;

	const int tensor_arena_size = 2 * 1024;
	uint8_t tensor_arena[tensor_arena_size];

	tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size, error_reporter);

	interpreter.AllocateTensors();

	TfLiteTensor* input = interpreter.input(0);

	TF_LITE_MICRO_EXPECT_NE(nullptr, input);
	TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
	TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
	TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
	TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);

	input->data.f[0] = 0.;

	TfLiteStatus invoke_status = interpreter.Invoke();
	if (invoke_status != kTfLiteOk) {
  		TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
	}

	TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

	TfLiteTensor* output = interpreter.output(0);
	TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
	TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
	TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
	TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

	float value = output->data.f[0];
	TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);
}

TF_LITE_MICRO_TESTS_END
