from base64 import b64encode
from django import template

register = template.Library()

@register.inclusion_tag('_comparison.html')
def _comparison(comparison, experiment):
    if comparison:
        left_mp4 = open(comparison.media_url_1,'rb').read()
        left_data_url = "data:video/mp4;base64," + b64encode(left_mp4).decode()

        right_mp4 = open(comparison.media_url_2,'rb').read()
        right_data_url = "data:video/mp4;base64," + b64encode(right_mp4).decode()
    else:
        left_data_url = None
        right_data_url = None
    return {'comparison': comparison, "experiment": experiment, 'left_data_url': left_data_url, 'right_data_url': right_data_url}
