from flask import Flask, request, jsonify
import requests
import psycopg2  

app = Flask(__name__)

DB_NAME = ai_assessment_engine
DB_USER = ai_ass_eng
DB_PASS = $TAkxA6uk#LBc0#d
API_VALID_TIME = 36000000
DB_HOST = placecom-co.cluster-cxjekraxhsam.ap-south-1.rds.amazonaws.com
DB_PORT = 3306

def get_data_from_db(batch_id):

    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        query = "SELECT * FROM batches WHERE batch_id = %s"
        cursor.execute(query, (batch_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        # Return the fetched data (you can customize this based on your DB structure)
        return result

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def post_data_to_server(server_url, data):
    """Function to post data to another server URL"""
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(server_url, json=data, headers=headers)
        return response
    except Exception as e:
        print(f"Error posting data: {e}")
        return None

@app.route('/process_batch', methods=['POST'])
def process_batch():
    """Endpoint to process the batch request"""
    try:
        # Get the JSON data from the request body
        request_data = request.get_json()

        batch_id = request_data.get('batch_id')
        server_url = request_data.get('server_url')

        if not batch_id or not server_url:
            return jsonify({"error": "Both 'batch_id' and 'server_url' are required"}), 400

        # Fetch data from DB based on batch_id
        data = get_data_from_db(batch_id)

        if data is None:
            return jsonify({"error": f"No data found for batch_id {batch_id}"}), 404

        # Post data to the server
        response = post_data_to_server(server_url, data)

        if response and response.status_code == 200:
            return jsonify({"message": "Data processed successfully", "response": response.json()}), 200
        else:
            return jsonify({"error": "Failed to post data to the server"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
