import streamlit as st
from utils import engineer_features, calculate_match_score
import pandas as pd
from utils import engineer_features, calculate_match_score
import pandas as pd


def recommend_students_for_job(job_details, students_df, weights, top_n=10):
    # Engineer features
    student_features, job_tfidf, tfidf = engineer_features(students_df, job_details)
    
    # Calculate scores for all students
    scores = []
    for i, student in students_df.iterrows():
        student_tfidf = student_features[i][:-1].reshape(1, -1)  # Exclude CGPA feature
        match_score = calculate_match_score(student, job_details, student_tfidf, job_tfidf, tfidf, weights)
        scores.append((student['student_id'], match_score))
    
    # Sort students by match score and get top N
    top_students = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    return top_students


def main():
    st.title("Student Recommendation App")
    st.write("Welcome to the Student Recommendation App!")

    # Input for job description
    job_description = st.text_area("Enter job description")

    # Input for skills
    skills = st.text_input("Enter required skills (keep it comma separated)")

    # Input for job position
    job_position = st.text_input("Enter job position")
    
    # Input for min cgpa
    min_cgpa = st.number_input("Enter minimum CGPA:", min_value=0.0, max_value=10.0, value=3.5)

    top_k_can = st.number_input("No. of recommendations:", min_value=1, value=5)

    # Convert skills input to a list
    weights = {
        'skills': 0.5,
        'cgpa': 0.2,
        'preferred_position': 0.3
    }
    job_details = {
        'job_position': job_position,
        'job_description': job_description,
        'skills': skills,
        'minimum_cgpa': min_cgpa  # This is optional, can be None
    }
    
    students_df = pd.read_csv("job_students.csv")
    students_df = students_df[["student_id", "job_positions", "skills", "education_performance", "work_experience_titles", "work_experience_descriptions"]]
    
    students_df.rename(columns={'job_positions': 'preferred_job_positions', 'education_performance': 'cgpa'}, inplace=True)
    if st.button("Submit"):
        if job_position and job_description and skills:
            recommended_students = recommend_students_for_job(job_details, students_df, weights, top_n=top_k_can)

            # Display results
            st.write(f"Top 10 Recommendations for {job_details['job_position']}:")
            st.write(f"Job Description: {job_details['job_description']}")
            st.write(f"Required Skills: {job_details['skills']}")
            if pd.notnull(job_details.get('minimum_cgpa')):
                st.write(f"Minimum CGPA: {job_details['minimum_cgpa']}")
            else:
                st.write("No minimum CGPA specified")
            st.write("\nTop recommended students:")

            for rank, (student_id, score) in enumerate(recommended_students, 1):
                student = students_df[students_df['student_id'] == student_id].iloc[0]
                st.write(f"\n  {rank}. Student ID: {student_id} (Match Score: {score:.1f}/10)")
                st.write(f"     Skills: {student['skills']}")
                if pd.notnull(student['cgpa']):
                    st.write(f"     CGPA: {student['cgpa']}")
                if pd.notnull(student['preferred_job_positions']) and student['preferred_job_positions']:
                    st.write(f"     Preferred Job Positions: {student['preferred_job_positions']}")
                if pd.notnull(student['work_experience_titles']) and student['work_experience_titles']:
                    st.write(f"     Work Experience: {student['work_experience_titles']}")


if __name__ == "__main__":
    main()