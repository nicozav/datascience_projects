#%%
#Import libraries
import pandas as pd
import psycopg2 as ps
import time
import matplotlib.pyplot as plt
import credentials
import torch
import torch.nn as nn
import numpy as np

# Connect to database
def connect_to_db(host_name,dbname,port,username,password):
    try:
        conn = ps.connect(host=host_name,database=dbname,user=username,password=password,port=port)
        
    except ps.OperationalError as e:
        raise e
    
    else:
        print('Connected')
    
    return conn

# Main
def main():

    # Query and Model:
    # Database credentials
    host_name = credentials.host_name
    dbname  = credentials.dbname
    port = credentials.port
    username  = credentials.username
    password = credentials.password

    # Database connection and cursor for psycopg2
    conn = connect_to_db(host_name,dbname,port,username,password)

    # Query database
    query = pd.read_sql_query("""SELECT * FROM videos ORDER BY upload_date""", conn, index_col = None)

    time.sleep(1)

    # Insert results from query into dataframe
    df = pd.DataFrame(query,columns=['video_id','video_title','upload_date', 'view_count','like_count','favorite_count','comment_count'])

    # Convert query results into tensors for model
    X_numpy = df['view_count'].to_numpy()
    X = torch.from_numpy(X_numpy.astype(np.float32))
    X = X.view(X.shape[0],1)

    y_numpy = df['comment_count'].to_numpy()
    y = torch.from_numpy(y_numpy.astype(np.float32))
    y = y.view(y.shape[0],1)

    # Model
    n_samples, n_features = X.shape

    input_size = n_features
    output_size = 1
    model = nn.Linear(input_size, output_size)

    # Loss and Optimizer
    learning_rate = 0.1
    critterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

    # Training Loop
    num_epochs = 500

    for epoch in range(num_epochs):
        
        #Forward pass and loss
        y_predicted = model(X)
        loss = critterion(y_predicted, y)
        
        # Backward pass
        loss.backward()
        
        #Update
        optimizer.step()
        
        optimizer.zero_grad()
        
        if (epoch+1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
            
    # Plot Results
    y_predicted = model(X).detach().numpy()
    plt.plot(X_numpy,y_numpy,'ro')
    plt.plot(X_numpy, y_predicted, 'b')
    plt.show()
    
#%%
# Run main
if __name__ == '__main__':
    main()
# %%
