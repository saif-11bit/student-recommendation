import re
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def preprocess_text(raw):
    '''Case specific to be used with pandas apply method'''
    try:
        if pd.isna(raw) or raw == '':
            return ''
        # remove carriage returns and new lines
        raw = raw.replace('\r', '')
        raw = raw.replace('\n', '')
        
        # brackets appear in all instances
        raw = raw.replace('[', '')
        raw = raw.replace(']', '')
        raw = raw.replace(')', '')
        raw = raw.replace('(', '')
        
        # removing html tags
        clean_html = re.compile('<.*?>')
        clean_text = re.sub(clean_html, ' ', raw)
        
        # removing duplicate whitespace in between words
        clean_text = re.sub(" +", " ", clean_text) 
        
        # stripping first and last white space 
        clean_text = clean_text.strip()
        
        # commas had multiple spaces before and after in each instance
        clean_text = re.sub(" , ", ", ", clean_text) 
        
        # eliminating the extra comma after a period
        clean_text = clean_text.replace('.,', '.')
        clean_text = re.sub(r'nbsp\s*', ' ', clean_text)
        
        # using try and except due to Nan in the column
    except:
        clean_text = np.nan
    return ' '.join(str(clean_text).lower().split(','))



def engineer_features(students_df, job_details):
    tfidf = TfidfVectorizer(token_pattern=r'\S+')
    
    # Combine all text fields for students
    student_text = (students_df['skills'].fillna('') + ' ' + 
                    students_df['preferred_job_positions'].fillna('') + ' ' + 
                    students_df['work_experience_titles'].fillna('') + ' ' + 
                    students_df['work_experience_descriptions'].fillna(''))
    student_text = student_text.apply(preprocess_text)
    
    # Prepare job text
    job_text = preprocess_text(job_details['job_position'] + ' ' +  job_details['job_description'] + ' ' +  job_details['skills'])
    
    all_text = pd.concat([student_text, pd.Series([job_text])])
    tfidf_matrix = tfidf.fit_transform(all_text)
    
    student_tfidf = tfidf_matrix[:-1]  # All but last row
    job_tfidf = tfidf_matrix[-1]  # Last row
    
    # CGPA features
    scaler = MinMaxScaler()
    student_cgpa = students_df['cgpa'].fillna(students_df['cgpa'].mean())  # Fill NaN with mean
    student_cgpa = scaler.fit_transform(student_cgpa.values.reshape(-1, 1))
    
    # Combine features
    student_features = np.hstack([student_tfidf.toarray(), student_cgpa])
    
    return student_features, job_tfidf, tfidf




def calculate_match_score(student, job_details, student_tfidf, job_tfidf, tfidf, base_weights):
    weights = base_weights.copy()
    score = 0
    
    # Skill and experience match (text similarity)
    skill_similarity = cosine_similarity(student_tfidf, job_tfidf)[0][0]
    score += weights['skills'] * skill_similarity
    
    # CGPA match
    if pd.notnull(student['cgpa']):
        if 'minimum_cgpa' in job_details and pd.notnull(job_details['minimum_cgpa']):
            # If minimum CGPA is specified, use binary match
            cgpa_match = int(student['cgpa'] >= job_details['minimum_cgpa'])
            score += weights['cgpa'] * cgpa_match
        else:
            # If no minimum CGPA is specified, use scaled CGPA score
            max_cgpa = 4.0  # Assuming 4.0 scale, adjust if necessary
            cgpa_score = student['cgpa'] / max_cgpa
            score += weights['cgpa'] * cgpa_score
    else:
        # If student CGPA is not available, redistribute its weight to skills
        weights['skills'] += weights['cgpa']
        weights['cgpa'] = 0
    
    # Preferred job position match
    if pd.notnull(student['preferred_job_positions']) and student['preferred_job_positions']:
        preferred_positions = set(student['preferred_job_positions'].lower().split(','))
        if job_details['job_position'].lower() in preferred_positions:
            score += weights['preferred_position']
    else:
        # If preferred positions are not specified, redistribute the weight
        weights['skills'] += weights['preferred_position']
        weights['preferred_position'] = 0
    
    # Normalize score and scale to 0-10
    total_weight = sum(weights.values())
    normalized_score = score / total_weight
    return normalized_score * 10