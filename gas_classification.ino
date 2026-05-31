/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/* Includes ---------------------------------------------------------------- */
#include <Gas_Classification_inferencing.h>

static const float samples[][32] = {

    // ACETONE
    {
        0.5206, 0.5447, 0.5687, 0.6125, 0.6564, 0.6935, 0.7311, 0.7596,
        0.7873, 0.7823, 0.7772, 0.7413, 0.7048, 0.6748, 0.6452, 0.6158,
        0.5863, 0.5687, 0.5509, 0.5372, 0.5234, 0.5072, 0.4913, 0.4766,
        0.4621, 0.4476, 0.4332, 0.4261, 0.4190, 0.4134, 0.4076, 0.7873
    },

    // ETHANOL
    {
        0.3338, 0.3562, 0.3791, 0.3970, 0.4150, 0.4550, 0.4981, 0.5240,
        0.5497, 0.5538, 0.5584, 0.5503, 0.5407, 0.5093, 0.4800, 0.4626,
        0.4449, 0.4292, 0.4127, 0.3999, 0.3865, 0.3738, 0.3613, 0.3479,
        0.3346, 0.3249, 0.3147, 0.3059, 0.2969, 0.2893, 0.2812, 0.5497
    },

    // AMMONIA
    {
        0.5549, 0.5661, 0.5784, 0.6246, 0.6711, 0.7151, 0.7587, 0.7843,
        0.8101, 0.8160, 0.8218, 0.7928, 0.7647, 0.7349, 0.7046, 0.6702,
        0.6355, 0.6162, 0.5965, 0.5797, 0.5633, 0.5533, 0.5435, 0.5312,
        0.5190, 0.5097, 0.5005, 0.4915, 0.4824, 0.4756, 0.4688, 0.8101
    },

    // TOLUOL
    {
        0.6673, 0.6961, 0.7249, 0.7617, 0.7983, 0.8494, 0.9007, 0.9322,
        0.9639, 0.9572, 0.9505, 0.9063, 0.8623, 0.8213, 0.7802, 0.7526,
        0.7248, 0.7023, 0.6797, 0.6633, 0.6470, 0.6341, 0.6211, 0.6046,
        0.5885, 0.5793, 0.5702, 0.5615, 0.5527, 0.5451, 0.5376, 0.9639
    }
};

const char* expected_labels[] = {
    "acetone",
    "ethanol",
    "ammonia",
    "toluol"
};

float features[32];

int current_sample = 0;

/**
 * @brief      Copy raw feature data in out_ptr
 *             Function called by inference library
 *
 * @param[in]  offset   The offset
 * @param[in]  length   The length
 * @param      out_ptr  The out pointer
 *
 * @return     0
 */
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

void print_inference_result(ei_impulse_result_t result);

/**
 * @brief      Arduino setup function
 */
void setup()
{
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");
}

/**
 * @brief      Arduino main function
 */
void loop()
{
    memcpy(features,
       samples[current_sample],
       sizeof(features));

    Serial.println("\n========================");
    Serial.print("EXPECTED: ");
    Serial.println(expected_labels[current_sample]);
    Serial.println("========================");
    
    ei_printf("Edge Impulse standalone inferencing (Arduino)\n");

    if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        ei_printf("The size of your 'features' array is not correct. Expected %lu items, but had %lu\n",
            EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
        delay(1000);
        return;
    }

    ei_impulse_result_t result = { 0 };

    // the features are stored into flash, and we don't want to load everything into RAM
    signal_t features_signal;
    features_signal.total_length = sizeof(features) / sizeof(features[0]);
    features_signal.get_data = &raw_feature_get_data;

    // invoke the impulse
    EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false /* debug */);
    if (res != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", res);
        return;
    }

    // print inference return code
    ei_printf("run_classifier returned: %d\r\n", res);
    print_inference_result(result);

    current_sample++;

    if (current_sample >= 4) {

        Serial.println("\nALL SAMPLES PROCESSED");

        while (true) {
            delay(1000);
        }
    }

    delay(2000);
}

void print_inference_result(ei_impulse_result_t result) {

    
    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }

    // Print how long it took to perform inference
    ei_printf("Inference time: %d ms\r\n",
        result.timing.classification);
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

#if EI_CLASSIFIER_HAS_VISUAL_ANOMALY
    ei_printf("Visual anomalies:\r\n");
    for (uint32_t i = 0; i < result.visual_ad_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.visual_ad_grid_cells[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }
#endif

}