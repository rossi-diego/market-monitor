"""
Email utilities for sending reports via Outlook.
Uses win32com for direct Outlook integration on Windows.
Embeds Plotly charts as static images directly in email body using CID.
"""
import tempfile
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import copy

import plotly.graph_objects as go
import pandas as pd


# ============================================================
# Configuration Classes
# ============================================================
@dataclass
class EmailConfig:
    """Email configuration settings."""
    # Recipients (can be a single email or list of emails)
    recipients: list[str] = field(default_factory=lambda: [
        "diego.santanna@oleoplan.com.br",
        "otavio.kucharski@oleoplan.com.br"
    ])

    # Email content
    subject: str = "Market Monitor - Relatório de Ratios"
    body_text: str = (
        "Segue abaixo o relatório atualizado com as análises das relações de "
        "preço de commodities considerando o último settlement price. As relações "
        "são construídas após conversão das unidades dos ativos para toneladas, "
        "sempre utilizando o continuation future 1"
    )
    footer_text: str = "Este email foi gerado automaticamente pelo Market Monitor Panel."

    # Chart styling for email
    theme: str = "plotly_dark"  # "plotly" for light, "plotly_dark" for dark
    background_color: str = "#111111"
    plot_background_color: str = "#1a1a1a"
    text_color: str = "#ffffff"
    grid_color: str = "rgba(255, 255, 255, 0.1)"
    line_color: str = "#555555"

    # Image export settings
    image_width: int = 1200
    image_height: int = 600

    # HTML styling
    html_font_family: str = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    html_primary_color: str = "#3498db"
    html_text_color: str = "#333"
    html_background_color: str = "#f9f9f9"

    # Display options
    show_timestamp: bool = True

    def get_recipients_string(self) -> str:
        """Convert recipients list to semicolon-separated string for Outlook."""
        if isinstance(self.recipients, list):
            return "; ".join(self.recipients)
        return self.recipients


# Default configuration instance
DEFAULT_CONFIG = EmailConfig()


# ============================================================
# Helper Functions
# ============================================================
def customize_figure_for_email(fig: go.Figure, config: EmailConfig = None) -> go.Figure:
    """
    Create a copy of the figure with styling optimized for email.
    Does not modify the original figure.

    Parameters
    ----------
    fig : go.Figure
        Original Plotly figure
    config : EmailConfig, optional
        Email configuration. Uses DEFAULT_CONFIG if not provided.

    Returns
    -------
    go.Figure
        Copy of figure with email styling applied
    """
    config = config or DEFAULT_CONFIG
    fig_email = copy.deepcopy(fig)

    # Apply email theme and colors
    fig_email.update_layout(
        template=config.theme,
        paper_bgcolor=config.background_color,
        plot_bgcolor=config.plot_background_color,
        font=dict(
            color=config.text_color,
            family="Arial, sans-serif",
        ),
        title=dict(
            font=dict(color=config.text_color)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=config.grid_color,
            zeroline=False,
            linecolor=config.line_color,
            tickfont=dict(color=config.text_color),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=config.grid_color,
            zeroline=False,
            linecolor=config.line_color,
            tickfont=dict(color=config.text_color),
        ),
    )

    # Update additional axes if they exist (for subplots)
    for i in range(2, 10):
        for axis_type in ["xaxis", "yaxis"]:
            axis_name = f"{axis_type}{i}"
            if hasattr(fig_email.layout, axis_name):
                getattr(fig_email.layout, axis_name).update(
                    showgrid=True,
                    gridcolor=config.grid_color,
                    zeroline=False,
                    linecolor=config.line_color,
                )

    return fig_email


def convert_charts_to_images(
    charts: Dict[str, go.Figure],
    config: EmailConfig = None
) -> Dict[str, Tuple[str, str]]:
    """
    Convert Plotly figures to PNG images and save to temporary directory.

    Parameters
    ----------
    charts : dict
        Dictionary of {chart_name: plotly_figure}
    config : EmailConfig, optional
        Email configuration. Uses DEFAULT_CONFIG if not provided.

    Returns
    -------
    dict
        Dictionary of {cid: (image_path, chart_name)}
    """
    config = config or DEFAULT_CONFIG

    # Create temporary directory
    temp_dir = Path(tempfile.gettempdir()) / "market_monitor_charts"
    temp_dir.mkdir(exist_ok=True)

    image_paths = {}

    for idx, (chart_name, fig) in enumerate(charts.items(), 1):
        safe_name = chart_name.replace("/", "_").replace(" ", "_")
        img_path = temp_dir / f"{safe_name}.png"

        try:
            # Customize figure for email
            fig_email = customize_figure_for_email(fig, config)

            # Export as PNG
            fig_email.write_image(
                str(img_path),
                width=config.image_width,
                height=config.image_height
            )
            image_paths[f"chart_{idx}"] = (str(img_path), chart_name)
            print(f"✅ Gráfico '{chart_name}' convertido para imagem")
        except Exception as e:
            print(f"⚠️  Aviso: Não foi possível converter '{chart_name}' para imagem")
            print(f"   Erro: {str(e)}")

    return image_paths


def build_html_body(
    body_text: str,
    image_paths: Dict[str, Tuple[str, str]],
    config: EmailConfig = None
) -> str:
    """
    Build HTML email body with embedded images.

    Parameters
    ----------
    body_text : str
        Main text content of the email
    image_paths : dict
        Dictionary of {cid: (image_path, chart_name)}
    config : EmailConfig, optional
        Email configuration. Uses DEFAULT_CONFIG if not provided.

    Returns
    -------
    str
        Complete HTML body string
    """
    config = config or DEFAULT_CONFIG
    timestamp = pd.Timestamp.now().strftime('%d/%m/%Y às %H:%M:%S')

    # Format body text with line breaks
    formatted_body = body_text.replace('\n', '<br>')

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: {config.html_font_family};
                color: {config.html_text_color};
                margin: 0;
                padding: 20px;
                background-color: {config.html_background_color};
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .header {{
                margin-bottom: 30px;
                border-bottom: 3px solid {config.html_primary_color};
                padding-bottom: 20px;
            }}
            .header p {{
                margin: 10px 0;
                font-size: 14px;
            }}
            .chart-section {{
                margin: 40px 0;
                page-break-inside: avoid;
                text-align: center;
            }}
            .chart-title {{
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 15px;
                color: #2c3e50;
                text-align: left;
            }}
            .chart-image {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin: 15px 0;
                display: block;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                font-size: 12px;
                color: #7f8c8d;
                text-align: center;
            }}
            .timestamp {{
                font-size: 13px;
                color: #95a5a6;
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <p><strong>{formatted_body}</strong></p>
                {f'<p class="timestamp">📅 Data: {timestamp}</p>' if config.show_timestamp else ''}
            </div>
    """

    # Add chart sections
    for cid, (img_path, chart_name) in image_paths.items():
        html += f"""
            <div class="chart-section">
                <div class="chart-title">📊 {chart_name}</div>
                <img src="cid:{cid}" class="chart-image" alt="{chart_name}">
            </div>
        """

    # Add footer
    html += f"""
            <div class="footer">
                <p>{config.footer_text}</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html


def cleanup_temp_files(image_paths: Dict[str, Tuple[str, str]]) -> None:
    """
    Remove temporary image files.

    Parameters
    ----------
    image_paths : dict
        Dictionary of {cid: (image_path, chart_name)}
    """
    for cid, (img_path, chart_name) in image_paths.items():
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
                print(f"🗑️  Arquivo temporário removido: {chart_name}")
            except Exception as e:
                print(f"⚠️  Aviso: Não foi possível remover {img_path}: {e}")


# ============================================================
# Main Email Function
# ============================================================
def send_email_with_embedded_charts(
    recipient: Optional[str] = None,
    charts: Optional[Dict[str, go.Figure]] = None,
    subject: Optional[str] = None,
    body_text: Optional[str] = None,
    config: EmailConfig = None,
) -> bool:
    """
    Send email via Outlook with embedded Plotly charts as PNG images.
    Images are saved temporarily, attached with CID, and deleted after sending.

    Parameters
    ----------
    recipient : str, optional
        Email recipient(s). If None, uses config.recipients
    charts : dict, optional
        Dictionary of {name: plotly_figure}
    subject : str, optional
        Email subject. If None, uses config.subject
    body_text : str, optional
        Email body text. If None, uses config.body_text
    config : EmailConfig, optional
        Email configuration. Uses DEFAULT_CONFIG if not provided.

    Returns
    -------
    bool
        True if sent successfully, False otherwise
    """
    config = config or DEFAULT_CONFIG

    # Use defaults from config if not provided
    recipient = recipient or config.get_recipients_string()
    subject = subject or config.subject
    body_text = body_text or config.body_text

    # Validate dependencies
    try:
        import win32com.client as win32
    except ImportError:
        print("❌ Erro: pywin32 não instalado. Execute: pip install pywin32")
        return False

    # Validate charts
    if not charts:
        print("❌ Erro: Nenhum gráfico fornecido")
        return False

    try:
        # Convert charts to images
        image_paths = convert_charts_to_images(charts, config)

        if not image_paths:
            print("❌ Erro: Nenhum gráfico foi convertido com sucesso")
            return False

        # Initialize Outlook
        outlook = win32.Dispatch('outlook.application')
        mail = outlook.CreateItem(0)

        # Configure email
        mail.To = recipient
        mail.Subject = subject

        # Build HTML body
        html_body = build_html_body(body_text, image_paths, config)
        mail.HTMLBody = html_body

        # Attach images with CID
        for cid, (img_path, chart_name) in image_paths.items():
            attachment = mail.Attachments.Add(os.path.abspath(img_path))
            # Set CID property for inline display
            attachment.PropertyAccessor.SetProperty(
                "http://schemas.microsoft.com/mapi/proptag/0x3712001E",
                cid
            )

        # Send email
        mail.Send()
        print(f"\n✅ Email enviado com sucesso para {recipient}")
        print(f"   📊 {len(image_paths)} gráfico(s) embutido(s) no corpo")

        # Cleanup temporary files
        cleanup_temp_files(image_paths)

        return True

    except Exception as e:
        print(f"❌ Erro ao enviar email: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# Alias for backward compatibility
send_email_with_chart_attachments = send_email_with_embedded_charts
