"""PDF Report Generator for Emotion Analysis Sessions."""

import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# Emotion colors matching the UI
EMOTION_COLORS = {
    'neutral': colors.HexColor('#8e8e93'),
    'calm': colors.HexColor('#5ac8fa'),
    'happy': colors.HexColor('#ffcc00'),
    'sad': colors.HexColor('#5856d6'),
    'angry': colors.HexColor('#ff3b30'),
    'fearful': colors.HexColor('#af52de'),
    'disgust': colors.HexColor('#34c759'),
    'surprised': colors.HexColor('#ff9500')
}

EMOTION_EMOJIS = {
    'neutral': '😐', 'calm': '😌', 'happy': '😊', 'sad': '😢',
    'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
}


class EmotionReportGenerator:
    """Generate PDF reports for emotion analysis sessions."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='Title',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1c1c1e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#8e8e93')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#0a84ff')
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            textColor=colors.HexColor('#1c1c1e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='StatLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#8e8e93')
        ))
        
        self.styles.add(ParagraphStyle(
            name='StatValue',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#1c1c1e')
        ))
    
    def generate_report(self, session_data: dict) -> bytes:
        """Generate a PDF report from session data.
        
        Args:
            session_data: Dictionary containing:
                - session_id: Unique session identifier
                - start_time: Session start timestamp
                - end_time: Session end timestamp
                - emotions: List of emotion detections [{emotion, confidence, timestamp}]
                - dominant_emotion: Most frequent emotion
                - average_confidence: Average confidence score
                - model_used: Model type (cnn, cnn_lstm, lstm)
                - language: Language used
                - transcriptions: List of transcriptions (optional)
        
        Returns:
            PDF file as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Emotion Analysis Report", self.styles['Title']))
        
        # Subtitle with date
        date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"Generated on {date_str}", self.styles['Subtitle']))
        
        story.append(Spacer(1, 20))
        
        # Session Overview
        story.append(Paragraph("Session Overview", self.styles['SectionHeader']))
        
        overview_data = [
            ['Session Duration', self._format_duration(session_data)],
            ['Model Used', session_data.get('model_used', 'CNN').upper()],
            ['Language', self._get_language_name(session_data.get('language', 'en'))],
            ['Total Detections', str(len(session_data.get('emotions', [])))],
        ]
        
        overview_table = Table(overview_data, colWidths=[2*inch, 4*inch])
        overview_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#8e8e93')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1c1c1e')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(overview_table)
        
        story.append(Spacer(1, 30))
        
        # Key Metrics
        story.append(Paragraph("Key Metrics", self.styles['SectionHeader']))
        
        dominant = session_data.get('dominant_emotion', 'neutral')
        avg_conf = session_data.get('average_confidence', 0.5) * 100
        
        metrics_data = [
            [
                self._create_metric_cell("Dominant Emotion", f"{EMOTION_EMOJIS.get(dominant, '😐')} {dominant.capitalize()}"),
                self._create_metric_cell("Average Confidence", f"{avg_conf:.0f}%"),
                self._create_metric_cell("Emotion Variety", str(len(self._get_emotion_breakdown(session_data))))
            ]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
        metrics_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (0, 0), 1, colors.HexColor('#e5e5ea')),
            ('BOX', (1, 0), (1, 0), 1, colors.HexColor('#e5e5ea')),
            ('BOX', (2, 0), (2, 0), 1, colors.HexColor('#e5e5ea')),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f7')),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ]))
        story.append(metrics_table)
        
        story.append(Spacer(1, 30))
        
        # Emotion Breakdown
        story.append(Paragraph("Emotion Breakdown", self.styles['SectionHeader']))
        
        breakdown = self._get_emotion_breakdown(session_data)
        if breakdown:
            # Create pie chart
            pie_chart = self._create_pie_chart(breakdown)
            story.append(pie_chart)
            
            story.append(Spacer(1, 20))
            
            # Breakdown table
            total = sum(breakdown.values())
            breakdown_data = [['Emotion', 'Count', 'Percentage']]
            for emotion, count in sorted(breakdown.items(), key=lambda x: -x[1]):
                pct = (count / total * 100) if total > 0 else 0
                breakdown_data.append([
                    f"{EMOTION_EMOJIS.get(emotion, '😐')} {emotion.capitalize()}",
                    str(count),
                    f"{pct:.1f}%"
                ])
            
            breakdown_table = Table(breakdown_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            breakdown_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0a84ff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e5ea')),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(breakdown_table)
        
        story.append(Spacer(1, 30))
        
        # Timeline (if emotions available)
        emotions = session_data.get('emotions', [])
        if emotions:
            story.append(Paragraph("Emotion Timeline", self.styles['SectionHeader']))
            
            timeline_data = [['Time', 'Emotion', 'Confidence']]
            for i, e in enumerate(emotions[-15:]):  # Last 15 entries
                time_str = e.get('timestamp', '')
                if time_str:
                    try:
                        dt = datetime.fromisoformat(time_str)
                        time_str = dt.strftime("%H:%M:%S")
                    except:
                        time_str = f"#{i+1}"
                else:
                    time_str = f"#{i+1}"
                
                timeline_data.append([
                    time_str,
                    f"{EMOTION_EMOJIS.get(e['emotion'], '😐')} {e['emotion'].capitalize()}",
                    f"{e['confidence']*100:.0f}%"
                ])
            
            timeline_table = Table(timeline_data, colWidths=[2*inch, 2.5*inch, 1.5*inch])
            timeline_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34c759')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e5ea')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f7')]),
            ]))
            story.append(timeline_table)
        
        story.append(Spacer(1, 30))
        
        # Insights
        story.append(Paragraph("Insights", self.styles['SectionHeader']))
        insights = self._generate_insights(session_data)
        for insight in insights:
            story.append(Paragraph(f"• {insight}", self.styles['BodyText']))
        
        story.append(Spacer(1, 40))
        
        # Footer
        footer_text = "Generated by Voice Emotion - Speech Emotion Recognition System"
        story.append(Paragraph(footer_text, self.styles['Subtitle']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _format_duration(self, session_data: dict) -> str:
        """Format session duration."""
        emotions = session_data.get('emotions', [])
        if len(emotions) < 2:
            return "< 1 minute"
        
        try:
            start = datetime.fromisoformat(emotions[0].get('timestamp', ''))
            end = datetime.fromisoformat(emotions[-1].get('timestamp', ''))
            duration = (end - start).total_seconds()
            
            if duration < 60:
                return f"{int(duration)} seconds"
            elif duration < 3600:
                return f"{int(duration // 60)} minutes"
            else:
                return f"{int(duration // 3600)} hours {int((duration % 3600) // 60)} minutes"
        except:
            return f"~{len(emotions) * 2} seconds"
    
    def _get_language_name(self, code: str) -> str:
        """Get language name from code."""
        languages = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'zh': 'Chinese', 'ja': 'Japanese',
            'ko': 'Korean', 'hi': 'Hindi', 'ar': 'Arabic', 'ru': 'Russian'
        }
        return languages.get(code, 'English')
    
    def _get_emotion_breakdown(self, session_data: dict) -> dict:
        """Get emotion counts from session data."""
        breakdown = session_data.get('emotion_breakdown', {})
        if breakdown:
            return breakdown
        
        emotions = session_data.get('emotions', [])
        counts = {}
        for e in emotions:
            em = e.get('emotion', 'neutral')
            counts[em] = counts.get(em, 0) + 1
        return counts
    
    def _create_metric_cell(self, label: str, value: str) -> str:
        """Create a metric cell content."""
        return f"<para align='center'><font size='10' color='#8e8e93'>{label}</font><br/><font size='16' color='#1c1c1e'><b>{value}</b></font></para>"
    
    def _create_pie_chart(self, breakdown: dict) -> Drawing:
        """Create a pie chart for emotion breakdown."""
        drawing = Drawing(400, 200)
        
        pie = Pie()
        pie.x = 100
        pie.y = 25
        pie.width = 150
        pie.height = 150
        
        pie.data = list(breakdown.values())
        pie.labels = [f"{e.capitalize()}" for e in breakdown.keys()]
        
        # Set colors
        for i, emotion in enumerate(breakdown.keys()):
            pie.slices[i].fillColor = EMOTION_COLORS.get(emotion, colors.gray)
            pie.slices[i].strokeColor = colors.white
            pie.slices[i].strokeWidth = 2
        
        pie.slices.strokeWidth = 1
        pie.slices.popout = 3
        
        drawing.add(pie)
        return drawing
    
    def _generate_insights(self, session_data: dict) -> list:
        """Generate insights from session data."""
        insights = []
        
        dominant = session_data.get('dominant_emotion', 'neutral')
        avg_conf = session_data.get('average_confidence', 0.5)
        breakdown = self._get_emotion_breakdown(session_data)
        total = sum(breakdown.values())
        
        # Dominant emotion insight
        if dominant in ['happy', 'calm']:
            insights.append(f"Your dominant emotion was {dominant}, indicating a positive emotional state during this session.")
        elif dominant in ['sad', 'fearful']:
            insights.append(f"Your dominant emotion was {dominant}. Consider activities that help improve your mood.")
        elif dominant == 'angry':
            insights.append("Anger was detected frequently. Deep breathing exercises may help manage stress.")
        else:
            insights.append(f"Your emotional state was primarily {dominant} during this session.")
        
        # Confidence insight
        if avg_conf > 0.7:
            insights.append("The emotion detection confidence was high, indicating clear emotional expressions.")
        elif avg_conf < 0.4:
            insights.append("Detection confidence was lower than usual. This could indicate subtle emotional expressions.")
        
        # Variety insight
        if len(breakdown) >= 4:
            insights.append("You displayed a wide range of emotions, showing emotional expressiveness.")
        elif len(breakdown) == 1:
            insights.append("Your emotional state remained consistent throughout the session.")
        
        # Specific patterns
        if 'happy' in breakdown and 'sad' in breakdown:
            happy_pct = breakdown.get('happy', 0) / total * 100
            sad_pct = breakdown.get('sad', 0) / total * 100
            if abs(happy_pct - sad_pct) < 10:
                insights.append("You showed a balance between positive and negative emotions.")
        
        return insights


# Singleton instance
_report_generator = None

def get_report_generator() -> EmotionReportGenerator:
    """Get or create the report generator singleton."""
    global _report_generator
    if _report_generator is None:
        _report_generator = EmotionReportGenerator()
    return _report_generator
