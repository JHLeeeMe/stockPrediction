# stockPrediction
YAHOO FINANCE에서 제공되는 amazon 주가 데이터(csv file)을 RNN model에 학습시켜 다음날의 종가를 예측  
![amazon_finance](https://user-images.githubusercontent.com/35649392/45922985-6b955500-bf14-11e8-9c93-828cb5e41e72.jpg)
<br>

## requirements
    tensorflow
    numpy
    pandas
<br>

## model graph
지난 7일간의 주식 정보를 이용해 다음날의 종가 예측 (MANY-TO-ONE)

![rnn_manytoone](https://user-images.githubusercontent.com/35649392/45923021-5a991380-bf15-11e8-9aae-4f56cf40b331.jpg)
<br>

## LSTM model
    # LSTM model

    def lstm_cell():
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
    
        if not keep_prob == 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) # dropOut
        
    return cell


    cells = [lstm_cell() for _ in range(stacked_layers_cnt)]
    if stacked_layers_cnt > 1:
        stacked_cells = tf.contrib.rnn.MultiRNNCell(cells)
    else:
        stacked_cells = lstm_cell()
    
    hypothesis, _states = tf.nn.dynamic_rnn(stacked_cells, X, dtype=tf.float32)
    hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt)    # MANY-TO-ONE

## cost & chart
![stock_prediction_pyplot](https://user-images.githubusercontent.com/35649392/45922919-9337ed80-bf13-11e8-9573-91e63c0f1b6d.jpg)
<br>
![stock_prediction_cost](https://user-images.githubusercontent.com/35649392/45922855-19533480-bf12-11e8-895f-ce10520f256a.jpg)
<br>
![stock_prediction_chart](https://user-images.githubusercontent.com/35649392/45922861-2ec85e80-bf12-11e8-91d2-8ec5a3a08cc0.jpg)
<br>
파란선이 실제, 빨간선이 예측  
경향을 잘 따라간다.

## prediction
![prediction](https://user-images.githubusercontent.com/35649392/45922678-c5465100-bf0d-11e8-834d-d5c3827b5bf3.jpg)
![amzn_finance](https://user-images.githubusercontent.com/35649392/45922677-c37c8d80-bf0d-11e8-96a0-fb3c3c0eba41.jpg)
