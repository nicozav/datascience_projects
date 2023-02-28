'''
YouTub API Project:

1. Learn to set up API calls and configure API parameters to request correct information
2. Learn to upload data to an RDS PostgreSQL database on AWS
3. YouTube Channel: SkillUp

'''

# Import all required libraries
import requests
import pandas as pd
import time
import psycopg2 as ps
import credentials

# API_KEY and CHANNEL_ID
API_KEY = credentials.API_KEY
CHANNEL_ID = credentials.CHANNEL_ID

# Functions for API request

# Second API request to collect video details based on video_id
def video_details(video_id):

    url_video_stats = 'https://www.googleapis.com/youtube/v3/videos?id='+video_id+'&part=statistics&key='+API_KEY
    response_video_stats = requests.get(url_video_stats).json()

    view_count = response_video_stats['items'][0]['statistics']['viewCount']
    like_count = response_video_stats['items'][0]['statistics']['likeCount']
    favorite_count = response_video_stats['items'][0]['statistics']['favoriteCount']
    comment_count = response_video_stats['items'][0]['statistics']['commentCount']
    
    return view_count,like_count,favorite_count,comment_count

# Main API request and append rows into dataframe
def get_videos(data):
    
    pageToken = ''
    
    while 1:
        
        url = 'https://www.googleapis.com/youtube/v3/search?key='+API_KEY+"&channelId="+CHANNEL_ID+"&part=snippet,id&order=date&maxResults=10000&"+pageToken
        
        response = requests.get(url).json()
        time.sleep(1)
        
        for video in response['items']:
            if video['id']['kind'] == 'youtube#video':
                video_id = video['id']['videoId']
                video_title = video['snippet']['title']
                video_title = str(video_title).replace('&#39;','').replace('&quot;','"')
                upload_date = video['snippet']['publishedAt']
                upload_date = str(upload_date).split('T')[0]
                
                # Execute second API call to get video details
                view_count,like_count,favorite_count,comment_count = video_details(video_id)

                # Saving data into pandas dataframe
                data = data.append({'video_id': video_id,'video_title':video_title,
                                    'upload_date':upload_date,'view_count':view_count,
                                    'like_count':like_count,'favorite_count':favorite_count,
                                'comment_count':comment_count}, ignore_index = True)
                 
        # Checking to see if there are more pages with more resutls       
        try:
            if response['nextPageToken'] != None: #if none, it means it reached the last page and break out of it
                pageToken = 'pageToken=' + response['nextPageToken']
        except:
            break
                
    return data

# Functions for database upload and data upkeep:

# Connect to database
def connect_to_db(host_name,dbname,port,username,password):
    try:
        conn = ps.connect(host=host_name,database=dbname,user=username,password=password,port=port)
        
    except ps.OperationalError as e:
        raise e
    
    else:
        print('Connected')
    
    return conn

# Check if row exists
def check_if_video_exists(curr,video_id):
    query = ("""SELECT video_id FROM videos WHERE video_id = %s""")
    curr.execute(query, (video_id,))
    
    return curr.fetchone() is not None

# Create table in PostgreSQL database
def create_table(curr):
    create_table_command = (""" CREATE TABLE IF NOT EXISTS videos (
                    video_id VARCHAR(255) PRIMARY KEY,
                    video_title TEXT NOT NULL,
                    upload_date DATE NOT NULL DEFAULT CURRENT_DATE,
                    view_count INTEGER NOT NULL,
                    like_count INTEGER NOT NULL,
                    favorite_count INTEGER NOT NULL,
                    comment_count INTEGER NOT NULL
            )""")
    
    curr.execute(create_table_command)

# Insert new row for video
def insert_into_table(curr, video_id, video_title, upload_date, view_count, like_count, favorite_count, comment_count):
    insert_into_videos = ("""INSERT INTO videos (video_id, video_title, upload_date,
                        view_count, like_count, favorite_count,comment_count)
    VALUES(%s,%s,%s,%s,%s,%s,%s);""")
    row_to_insert = (video_id, video_title, upload_date, view_count, like_count, favorite_count, comment_count)
    curr.execute(insert_into_videos, row_to_insert)

# Update row for video
def update_row(curr, video_id, video_title, view_count, like_count, favorite_count, comment_count):
    query = ("""
             UPDATE videos
             SET video_title = %s,
                view_count = %s,
                like_count = %s,
                favorite_count = %s,
                comment_count = %s
             WHERE video_id  = %s;""")
    vars_to_update = (video_title, view_count, like_count, favorite_count, comment_count, video_id)
    curr.execute(query, vars_to_update)
 
# Truncate table   
def truncate_table(curr):
    truncate_table = ("""TRUNCATE TABLE videos""")

    curr.execute(truncate_table)

# Append from dataframe to database
def append_from_df_to_db(curr,df):
    for i, row in df.iterrows():
        insert_into_table(curr, row['video_id'], row['video_title'], row['upload_date'], row['view_count']
                          , row['like_count'], row['favorite_count'], row['comment_count'])

# Create temp dataframe with updated info
def update_db(curr,df):
    tmp_df = pd.DataFrame(columns=['video_id', 'video_title', 'upload_date', 'view_count',
                                   'like_count', 'favorite_count', 'comment_count'])
    for i, row in df.iterrows():
        if check_if_video_exists(curr, row['video_id']): # If video already exists then we will update
            update_row(curr,row['video_id'],row['video_title'],row['view_count'],row['like_count'],row['favorite_count'],row['comment_count'])
        else: # The video doesn't exists so we will add it to a temp df and append it using append_from_df_to_db
            tmp_df = tmp_df.append(row)

    return tmp_df

# Main
def main():
    
    # Database credentials:
    host_name = credentials.host_name
    dbname  = credentials.dbname
    port = credentials.port
    username  = credentials.username
    password = credentials.password
    conn = None
    
    # Database connection and cursor for psycopg2
    conn = connect_to_db(host_name,dbname,port,username,password)
    curr = conn.cursor()
    
    time.sleep(1)
    
    # Create Table if doesn't exist
    create_table(curr)
    
    # Create dataframe structure
    df = pd.DataFrame(columns=['video_id','video_title','upload_date','view_count',
                        'like_count','favorite_count','comment_count'])
    # Execute YouTube API request and insert into dataframe
    df = get_videos(df)

    # Create new dataframe with updated data
    new_vid_df = update_db(curr,df)
    conn.commit()

    #Insert new videos into db table
    append_from_df_to_db(curr, new_vid_df)
    conn.commit()

# Execute Function:

if __name__ == '__main__':
    main()