import pandas as pd
import os

def chunk_story(df, chunk_size=1000, overlap_size=0):
    '''
    This function chunks the story into smaller pieces, using overlap if specified
    
    Args:
    df: DataFrame
    chunk_size: int
    overlap_size: int
    
    Returns:
    new_df: DataFrame
    
    '''
    # initialize an empty list to store the new rows
    new_rows = []

    for _, row in df.iterrows():
        champion_name = row['champion']
        story = row['story']
        
        i = 0
        while i < len(story):
            # if overlap_size is greater than 0, move the start of the chunk back to create an overlap
            if overlap_size > 0 and i != 0:
                i -= overlap_size  # move the start of the chunk back by the overlap size
                # if the overlap is too large, move the start of the chunk to the beginning of the story
                next_sentence_start = story[i:].find(". ") + 2  # find the next sentence
                if 0 < next_sentence_start < len(story[i:]):
                    i += next_sentence_start  # move the start of the chunk to the next sentence
            
            # find the end of the chunk
            chunk_end = i + chunk_size
            if chunk_end < len(story):
                last_period_idx = story[i:chunk_end].rfind(".")
                if last_period_idx != -1:
                    chunk = story[i:i + last_period_idx + 1]  # include the period in the chunk
                else:
                    # if there is no period in the chunk, find the next period after the chunk
                    chunk = story[i:chunk_end]
            else:
                chunk = story[i:]
            
            # add the new row to the list
            new_rows.append({"champion": champion_name, "story_chunk": chunk})
            i += len(chunk)  # move the start of the next chunk to the end of the current chunk
    
    # create a new DataFrame from the list of new rows
    new_df = pd.DataFrame(new_rows)
    return new_df

if __name__ == '__main__':
    # Load the CSV file into a pandas DataFrame
    def load_data():
        if os.path.exists('../data/champions_lore.csv'):
            df = pd.read_csv('../data/champions_lore.csv')
        else:
            print("File not found")
        return df
    df = load_data()
    # create a new DataFrame with the story chunks
    new_df = chunk_story(df, chunk_size=1000, overlap_size=200)