from flask import Flask, request, jsonify
from keras.models import load_model
import joblib
import numpy as np


CATEGORY_SIZE = 8
IPAddr = '**'   #<-------------Set your IP Address here

app = Flask(__name__)

loaded_model = load_model('SAVED_MODELS\KerasModel2000.h5')
scaler = joblib.load('SAVED_MODELS\scaler_save2000.joblib')

use_price_array = [0, 1, 0, 0] #price_method = 'cm' for all items

#sets baseline value
def new_ap_est(OCP, NCP, OAP):
    return NCP/OCP * OAP

@app.route('/', methods=['POST'])
def calculate_prices():
    try:
        # Parse the JSON body from the POST request
        request_data = request.get_json()

        print(request_data)

        # Check if the 'items' array is present in the request data
        if 'items' in request_data:
            # Initialize an array to store the response data
            response_data = {'items': []}
            
            # Process each item in the request
            for item in request_data['items']:
                ap_est = new_ap_est(item['old_cost_price'], item['new_cost_price'],item['old_agreement_price'])

                result_arrays = [item['old_cost_price'], item['new_cost_price'],item['old_agreement_price']]
                
                result_arrays = result_arrays +[ap_est]+ use_price_array

                print(len(result_arrays))
                print(result_arrays)

                model_input = np.array(result_arrays).reshape(1, len(result_arrays))

                Scaled_input = scaler.transform(model_input)
                prediction = loaded_model.predict(Scaled_input)
                print(int(prediction[0][0]))

                # Build the response for each item
                response_item = {
                    'item_obj': item['item_obj'],
                    'agreement_obj': item['agreement_obj'],
                    'new_agreement_price': f'{prediction[0][0]:.2f}'  # Format the result and return the value
                }

                # Add the response item to the array
                response_data['items'].append(response_item)

            # Send the response as JSON
            return jsonify(response_data)

        else:
            # If the 'items' array is not present, return an error response
            return jsonify({'error': 'Invalid request structure'}), 400

    except Exception as e:
        # Handle any unexpected errors
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host=IPAddr, port=4444)
