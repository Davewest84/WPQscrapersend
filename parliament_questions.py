#!/usr/bin/env python3
"""
UK Parliament Written Questions Analyzer
Fetches questions, analyzes with AI, and emails results
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import html
import sys

class ParliamentQuestionsAPI:
    """
    A class to interact with the UK Parliament Written Questions API
    """
    
    BASE_URL = "https://questions-statements-api.parliament.uk/api"
    
    def __init__(self):
        self.session = requests.Session()
        
    def get_questions(self, 
                     tabled_from=None, 
                     tabled_to=None,
                     answered=None,
                     answered_from=None,
                     answered_to=None,
                     search_term=None,
                     answering_bodies=None,
                     house=None,
                     expand_member=True,
                     take=100,
                     skip=0):
        """
        Get written questions with various filters
        """
        
        endpoint = f"{self.BASE_URL}/writtenquestions/questions"
        
        params = {
            'take': take,
            'skip': skip,
            'expandMember': expand_member
        }
        
        # Add optional parameters
        if tabled_from:
            params['tabledWhenFrom'] = tabled_from
        if tabled_to:
            params['tabledWhenTo'] = tabled_to
        if answered:
            params['answered'] = answered
        if answered_from:
            params['answeredWhenFrom'] = answered_from
        if answered_to:
            params['answeredWhenTo'] = answered_to
        if search_term:
            params['searchTerm'] = search_term
        if answering_bodies:
            params['answeringBodies'] = answering_bodies
        if house:
            params['house'] = house
            
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_all_questions(self, **kwargs):
        """
        Get all questions matching the criteria (handles pagination automatically)
        """
        all_results = []
        skip = 0
        take = 100  # Maximum per request
        
        while True:
            print(f"Fetching records {skip} to {skip + take}...")
            data = self.get_questions(skip=skip, take=take, **kwargs)
            
            if not data.get('results'):
                break
                
            all_results.extend(data['results'])
            
            # Check if we've retrieved all records
            if len(all_results) >= data.get('totalResults', 0):
                break
                
            skip += take
            time.sleep(0.5)  # Be polite to the API
            
        print(f"Total records retrieved: {len(all_results)}")
        return all_results
    
    def get_question_by_id(self, question_id, expand_member=True):
        """
        Get a single question by its ID with full details
        """
        endpoint = f"{self.BASE_URL}/writtenquestions/questions/{question_id}"
        
        params = {
            'expandMember': expand_member
        }
        
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()


class NewsStoryAnalyzer:
    """
    Analyzes parliamentary questions to identify newsworthy stories
    using an LLM API (Claude, OpenAI, etc.)
    """
    
    def __init__(self, api_key, provider='anthropic', model=None):
        """
        Initialize the analyzer with LLM credentials
        """
        self.api_key = api_key
        self.provider = provider.lower()
        
        if model:
            self.model = model
        else:
            self.model = 'claude-3-sonnet-20240229' if self.provider == 'anthropic' else 'gpt-4-turbo'
    
    def analyze_questions_for_newsworthiness(self, df, publication_examples, max_questions=None):
        """
        Analyze questions and identify newsworthy stories for a specific publication
        """
        
        # Limit the number of questions if specified
        questions_to_analyze = df.head(max_questions) if max_questions else df
        
        # Prepare the questions summary for the LLM
        questions_summary = self._prepare_questions_summary(questions_to_analyze)
        
        # Create the prompt
        prompt = self._create_analysis_prompt(questions_summary, publication_examples)
        
        print("Analyzing questions with LLM...")
        print("=" * 60)
        
        # Call the appropriate LLM API
        if self.provider == 'anthropic':
            response = self._call_anthropic_api(prompt)
        elif self.provider == 'openai':
            response = self._call_openai_api(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Parse the LLM response
        newsworthy_stories = self._parse_llm_response(response, df)
        
        return newsworthy_stories
    
    def _prepare_questions_summary(self, df):
        """Prepare a concise summary of questions for the LLM"""
        summaries = []
        
        for idx, row in df.iterrows():
            summary = f"""
Question ID: {row['id']}
UIN: {row['uin']}
Date Answered: {row['date_answered']}
Asking Member: {row['asking_member_name']} ({row['asking_member_party']})
Answering Body: {row['answering_body_name']}
Question: {row['question_text'][:500]}...
Answer: {row['answer_text'][:1000] if pd.notna(row['answer_text']) else 'No answer yet'}...
Attachments: {row['attachment_count']}
---"""
            summaries.append(summary)
        
        return '\n'.join(summaries)
    
    def _create_analysis_prompt(self, questions_summary, publication_examples):
        """Create the prompt for the LLM"""
        
        prompt = f"""You are a news editor analyzing parliamentary written questions to identify stories that would interest readers of a specific publication.

PUBLICATION CONTEXT:
Here are example headlines and subject lines from this publication that show their editorial style and interests:

{publication_examples}

TASK:
Analyze the following parliamentary questions and answers. Identify which ones would make newsworthy stories for this publication.

For each newsworthy question, provide:
1. Question ID
2. News headline (in the style of the publication)
3. News angle (why this matters to the publication's readers)
4. Story priority (High/Medium/Low)
5. Brief explanation (2-3 sentences)

PARLIAMENTARY QUESTIONS TO ANALYZE:

{questions_summary}

Please respond in the following JSON format:
{{
  "newsworthy_stories": [
    {{
      "question_id": 12345,
      "uin": "123456",
      "headline": "Proposed headline here",
      "news_angle": "Why this matters",
      "priority": "High|Medium|Low",
      "explanation": "Brief explanation of newsworthiness"
    }}
  ],
  "summary": "Overall summary of findings"
}}

Only include questions that would genuinely interest this publication's readers. Be selective and focus on real news value."""

        return prompt
    
    def _call_anthropic_api(self, prompt):
        """Call the Anthropic (Claude) API"""
        
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result['content'][0]['text']
        except requests.exceptions.HTTPError as e:
            print(f"API Error: {e}")
            print(f"Response: {response.text}")
            raise
    
    def _call_openai_api(self, prompt):
        """Call the OpenAI API"""
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a news editor analyzing parliamentary questions for newsworthiness."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def _parse_llm_response(self, response, original_df):
        """Parse the LLM's JSON response into a DataFrame"""
        
        try:
            # Extract JSON from response (handle markdown code blocks)
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            if response_clean.startswith('```'):
                response_clean = response_clean[3:]
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            
            response_clean = response_clean.strip()
            
            # Parse JSON
            parsed = json.loads(response_clean)
            
            # Convert to DataFrame
            if 'newsworthy_stories' in parsed and parsed['newsworthy_stories']:
                stories_df = pd.DataFrame(parsed['newsworthy_stories'])
                
                # Merge with original data to get full details
                stories_df = stories_df.merge(
                    original_df[['id', 'url', 'has_attachments', 'attachment_count', 'question_text', 'answer_text', 'asking_member_name', 'date_answered']],
                    left_on='question_id',
                    right_on='id',
                    how='left'
                )
                
                # Store the summary
                if 'summary' in parsed:
                    print(f"\n\nLLM Summary:\n{parsed['summary']}\n")
                
                return stories_df
            else:
                print("No newsworthy stories identified by the LLM")
                return pd.DataFrame()
                
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response}")
            return pd.DataFrame()


def questions_to_dataframe(questions_data, fetch_full_details=False, api_client=None):
    """
    Convert questions API response to a pandas DataFrame
    """
    
    records = []
    
    for idx, item in enumerate(questions_data):
        q = item.get('value', {})
        question_id = q.get('id')
        
        # If fetch_full_details is True, make an individual API call for this question
        if fetch_full_details and api_client and question_id:
            print(f"Fetching full details for question {idx + 1}/{len(questions_data)} (ID: {question_id})...")
            try:
                full_question_data = api_client.get_question_by_id(question_id)
                q = full_question_data.get('value', q)  # Use full data if available
                time.sleep(0.3)  # Be polite to the API
            except Exception as e:
                print(f"Warning: Could not fetch full details for question {question_id}: {e}")
        
        # Extract asking member info
        asking_member = q.get('askingMember', {})
        
        # Extract answering member info
        answering_member = q.get('answeringMember', {})
        
        # Extract attachments info
        attachments = q.get('attachments', [])
        attachment_count = len(attachments) if attachments else q.get('attachmentCount', 0)
        
        # Construct the URL for the question
        question_url = f"https://questions-statements.parliament.uk/written-questions/detail/{q.get('id')}" if q.get('id') else None
        
        # Flag for attachments - check both the list and the count
        has_attachments = 'YES' if (attachment_count > 0 or (attachments and len(attachments) > 0)) else 'NO'
        
        record = {
            'id': q.get('id'),
            'uin': q.get('uin'),
            'url': question_url,
            'has_attachments': has_attachments,
            'attachment_count': attachment_count,
            'date_tabled': q.get('dateTabled'),
            'date_for_answer': q.get('dateForAnswer'),
            'date_answered': q.get('dateAnswered'),
            'house': q.get('house'),
            'question_text': q.get('questionText'),
            'answer_text': q.get('answerText'),
            'answering_body_name': q.get('answeringBodyName'),
            'asking_member_name': asking_member.get('name'),
            'asking_member_party': asking_member.get('party'),
            'asking_member_from': asking_member.get('memberFrom'),
            'answering_member_name': answering_member.get('name'),
            'is_withdrawn': q.get('isWithdrawn'),
            'is_named_day': q.get('isNamedDay'),
            'heading': q.get('heading'),
            'attachments': str(attachments) if attachments else None,  # Convert to string for CSV export
        }
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Convert date columns to datetime
    date_columns = ['date_tabled', 'date_for_answer', 'date_answered']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df


def send_email_with_attachments(sender_email, sender_password, recipient_email, subject, body, attachments=[]):
    """
    Send an email with attachments using Gmail SMTP
    """
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Add body
        msg.attach(MIMEText(body, 'html'))
        
        # Add attachments
        for filepath in attachments:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(filepath)}'
                )
                msg.attach(part)
                print(f"  ✓ Attached: {os.path.basename(filepath)}")
            else:
                print(f"  ✗ File not found: {filepath}")
        
        # Connect to Gmail SMTP server
        print("\nConnecting to Gmail SMTP server...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        
        # Login
        print("Logging in...")
        server.login(sender_email, sender_password)
        
        # Send email
        print("Sending email...")
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        print(f"\n✅ Email sent successfully to {recipient_email}!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error sending email: {e}")
        return False


def main():
    """Main execution function"""
    
    # Configuration from environment variables
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    SENDER_EMAIL = os.environ.get('SENDER_EMAIL')
    SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD')
    RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', 'dave.west@hsj.co.uk')
    
    # Check required environment variables
    if not all([ANTHROPIC_API_KEY, SENDER_EMAIL, SENDER_PASSWORD]):
        print("❌ Missing required environment variables:")
        if not ANTHROPIC_API_KEY:
            print("  - ANTHROPIC_API_KEY")
        if not SENDER_EMAIL:
            print("  - SENDER_EMAIL")
        if not SENDER_PASSWORD:
            print("  - SENDER_PASSWORD")
        sys.exit(1)
    
    # Publication examples
    PUBLICATION_EXAMPLES = """
Example headlines from our publication:

- CQC chief executive announces sudden departure
- Two ICB chief executives to stand down
- NHSE imposes 'return to office' policy
- Revealed: The leaders shaping the 10-Year Health Plan
- Cut corporate services spend, trusts told
- 'No rush' to transfer NHSE staff as abolition faces delay
- NHSE and DHSC to be cut by 50%
- NHSE workforce bosses to leave
- Chris Hopson to leave NHSE
- NHS manager facing bribery charge found dead
- Trusts' 'league table' rankings revealed
- Revealed: The 10-Year Plan vision for FTs and ICBs
- We have seen the government's 10-Year Health Plan: it is a mess
- Embattled chief executive resigns
- Exclusive: NHSE director resigns, claiming politicians wanted 'change at the top'
- Go-ahead finally given for ICB and NHSE redundancies
- Trust ordered to cut 600 posts
- 'Big consolidation' of ICBs coming, says new NHSE chief exec
- ICB chief suspended pending internal investigation

Our publication focuses on: NHS leadership changes and appointments, organizational restructuring (ICBs, trusts, NHSE), 
workforce and staffing issues, budget cuts and financial pressures, policy changes affecting NHS management, 
internal investigations and governance issues, and the 10-Year Health Plan implementation.
We prioritize exclusive revelations, senior appointments/departures, and stories that impact NHS organizational structure.
"""
    
    print("=" * 60)
    print("UK PARLIAMENT WRITTEN QUESTIONS ANALYZER")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize API
    api = ParliamentQuestionsAPI()
    
    # Get questions from last 3 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    print("Retrieving questions answered in the last 3 days by Answering Body ID 17")
    print("=" * 60)
    
    questions = api.get_all_questions(
        answered_from=start_date.strftime('%Y-%m-%d'),
        answered_to=end_date.strftime('%Y-%m-%d'),
        answering_bodies=[17],
        answered='Answered'
    )
    
    # Convert to DataFrame with full details
    print("\n\nFetching full details for each question...")
    print("=" * 60)
    df_recent = questions_to_dataframe(questions, fetch_full_details=True, api_client=api)
    
    # Check if answers contain HTML tables
    df_recent['has_table'] = df_recent['answer_text'].apply(
        lambda x: 'YES' if (pd.notna(x) and ('<table' in str(x).lower() or '<tbody' in str(x).lower())) else 'NO'
    )
    
    # Sort by attachments, tables, then date
    df_recent['attachment_sort'] = df_recent['has_attachments'].map({'YES': 0, 'NO': 1})
    df_recent['table_sort'] = df_recent['has_table'].map({'YES': 0, 'NO': 1})
    df_recent = df_recent.sort_values(['attachment_sort', 'table_sort', 'date_answered'], ascending=[True, True, False])
    df_recent = df_recent.drop(['attachment_sort', 'table_sort'], axis=1)
    
    print(f"\nCompleted! Total questions with full details: {len(df_recent)}")
    print(f"Questions with attachments: {(df_recent['has_attachments'] == 'YES').sum()}")
    print(f"Questions with tables in answers: {(df_recent['has_table'] == 'YES').sum()}")
    
    # Export to CSV
    print("\n\nExporting to CSV")
    print("=" * 60)
    column_order = ['id', 'url', 'has_attachments', 'has_table', 'date_answered', 'heading', 'answer_text', 'question_text', 'asking_member_name', 'attachments']
    df_export = df_recent[column_order]
    df_export.to_csv('parliament_questions_body17_last7days.csv', index=False)
    print("Data exported to 'parliament_questions_body17_last7days.csv'")
    
    # LLM Analysis
    print("\n\n" + "=" * 60)
    print("LLM-POWERED NEWS STORY IDENTIFICATION")
    print("=" * 60)
    
    analyzer = NewsStoryAnalyzer(
        api_key=ANTHROPIC_API_KEY,
        provider='anthropic',
        model='claude-3-haiku-20240307'
    )
    
    newsworthy_df = analyzer.analyze_questions_for_newsworthiness(
        df=df_recent,
        publication_examples=PUBLICATION_EXAMPLES,
        max_questions=20
    )
    
    if not newsworthy_df.empty:
        newsworthy_df.to_csv('newsworthy_stories.csv', index=False)
        print("\n\nNewsworthy stories exported to 'newsworthy_stories.csv'")
    
    # Prepare and send email
    print("\n\n" + "=" * 60)
    print("PREPARING EMAIL")
    print("=" * 60)
    
    email_subject = f"Parliamentary Questions Report - {datetime.now().strftime('%Y-%m-%d')}"
    
    # Create HTML email body
    email_body = f"""
<html>
<head>
<style>
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 12px; }}
th {{ background-color: #4CAF50; color: white; padding: 8px; text-align: left; border: 1px solid #ddd; }}
td {{ padding: 8px; border: 1px solid #ddd; vertical-align: top; }}
tr:nth-child(even) {{ background-color: #f2f2f2; }}
.yes {{ background-color: #ffeb3b; font-weight: bold; }}
.table {{ background-color: #e3f2fd; font-weight: bold; }}
.priority-high {{ background-color: #f44336; color: white; padding: 2px 6px; border-radius: 3px; }}
.priority-medium {{ background-color: #ff9800; color: white; padding: 2px 6px; border-radius: 3px; }}
.priority-low {{ background-color: #2196F3; color: white; padding: 2px 6px; border-radius: 3px; }}
</style>
</head>
<body>
<h2>Parliamentary Questions Report</h2>
<p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>Answering Body:</strong> Department of Health and Social Care (ID: 17)</p>
<p><strong>Date Range:</strong> Last 7 days</p>

<h3>Summary</h3>
<ul>
<li><strong>Total Questions:</strong> {len(df_recent)}</li>
<li><strong>Questions with Attachments:</strong> {(df_recent['has_attachments'] == 'YES').sum()}</li>
<li><strong>Questions with Tables in Answers:</strong> {(df_recent['has_table'] == 'YES').sum()}</li>
"""
    
    if not newsworthy_df.empty:
        email_body += f"<li><strong>Newsworthy Stories Identified:</strong> {len(newsworthy_df)}</li>\n"
    
    email_body += "</ul>\n\n"
    
    # Add newsworthy stories
    if not newsworthy_df.empty:
        email_body += "<h3>🎯 Newsworthy Stories (AI Analysis)</h3>\n<table>\n"
        email_body += "<tr><th>Priority</th><th>Headline</th><th>ID</th><th>Attachments</th><th>News Angle</th><th>Explanation</th><th>Link</th></tr>\n"
        
        for idx, story in newsworthy_df.iterrows():
            priority_class = f"priority-{story['priority'].lower()}"
            attachment_class = 'class="yes"' if story['has_attachments'] == 'YES' else ''
            headline = html.escape(str(story['headline'])) if pd.notna(story['headline']) else ''
            news_angle = html.escape(str(story['news_angle'])) if pd.notna(story['news_angle']) else ''
            explanation = html.escape(str(story['explanation'])) if pd.notna(story['explanation']) else ''
            
            email_body += f"""
<tr>
<td><span class="{priority_class}">{story['priority']}</span></td>
<td><strong>{headline}</strong></td>
<td>{story['question_id']}</td>
<td {attachment_class}>{story['has_attachments']}</td>
<td>{news_angle}</td>
<td>{explanation}</td>
<td><a href="{story['url']}">View</a></td>
</tr>
"""
        email_body += "</table>\n\n"
    
    # Add full questions table
    email_body += "<h3>📋 All Questions (Full Details)</h3>\n<table>\n"
    email_body += "<tr><th>ID</th><th>Link</th><th>Attachments</th><th>Has Table</th><th>Date Answered</th><th>Heading</th><th>Question</th><th>Answer</th><th>Asked By</th></tr>\n"
    
    for idx, row in df_recent.iterrows():
        attachment_class = 'class="yes"' if row['has_attachments'] == 'YES' else ''
        table_class = 'class="table"' if row['has_table'] == 'YES' else ''
        
        question_text = html.escape(str(row['question_text']) if pd.notna(row['question_text']) else '')
        answer_text = html.escape(str(row['answer_text']) if pd.notna(row['answer_text']) else '')
        question_preview = (question_text[:200] + '...') if len(question_text) > 200 else question_text
        answer_preview = (answer_text[:300] + '...') if len(answer_text) > 300 else answer_text
        heading = html.escape(str(row['heading'])) if pd.notna(row['heading']) else ''
        asking_member = html.escape(str(row['asking_member_name'])) if pd.notna(row['asking_member_name']) else ''
        
        email_body += f"""
<tr>
<td>{row['id']}</td>
<td><a href="{row['url']}">View</a></td>
<td {attachment_class}>{row['has_attachments']}</td>
<td {table_class}>{row['has_table']}</td>
<td>{row['date_answered'].strftime('%Y-%m-%d') if pd.notna(row['date_answered']) else 'N/A'}</td>
<td>{heading}</td>
<td>{question_preview}</td>
<td>{answer_preview}</td>
<td>{asking_member}</td>
</tr>
"""
    
    email_body += "</table>\n\n"
    email_body += "<p style='color: #666; font-size: 11px; margin-top: 30px;'>This is an automated report generated from the UK Parliament Written Questions API.</p>"
    email_body += "</body></html>"
    
    # Send email
    attachments_list = ['parliament_questions_body17_last7days.csv']
    if not newsworthy_df.empty:
        attachments_list.append('newsworthy_stories.csv')
    
    send_email_with_attachments(
        sender_email=SENDER_EMAIL,
        sender_password=SENDER_PASSWORD,
        recipient_email=RECIPIENT_EMAIL,
        subject=email_subject,
        body=email_body,
        attachments=attachments_list
    )
    
    print("\n" + "=" * 60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
    
