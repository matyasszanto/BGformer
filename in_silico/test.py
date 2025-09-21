

import http.client
import json

import numpy as np

connection = http.client.HTTPConnection( 'localhost', port = 8008 )

headers = { 'Content-type': 'application/json; utf-8' }

data = {
        'SI': np.fromfile('out.csv', sep=',', dtype=float).tolist(),
        'glucose': [0.0, 0.0, 0.0, 0.0, 0.0], 
        'insulin': [5116.666666666667, 7116.666666666672, 7025.0000000000055, 1525.0, 1525.0], 
        'bgt': [0.0, 60.0, 180.0, 300.0], 
        'bg': [13.6, 11.592812687022564, 7.739154343059482, 6.901311652299331], 
        'gender': -1, 
        'age': 66.0
}
json_data = json.dumps( data )

print(f'read SI data is: {data["SI"]}')

connection.request( 'POST', '/', json_data, headers )
response = connection.getresponse()
print( 
        'Status: {} and reason: {}'.format(
                response.status, 
                response.reason
        )
)
# print( response )
print( response.read().decode() )

connection.close()





