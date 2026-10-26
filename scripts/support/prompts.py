"""Original prompt templates and the message builders used for cache replay."""

import hashlib
import json

from config import CONFIG
from .common import request

RETRIEVAL = ("You are a helpful news agent whose job is to select news articles that would help "
             "answer the user's query. Return a list of exactly 10 different article IDs.\n\n"
             "Here is a list of articles you can read:\n")
SYNTHESIS = ("You are a helpful news agent whose job is to answer the user's query "
             "based on the provided news articles.\n\n"
             "Here is the list of articles you can read: {articles}")
ANSWER_ANNOTATION = """You are an expert media analyst annotating how a text frames one event. You will see a list of frames, each as [frame_id] name: description, and one text.

For EVERY frame, score how salient it is in the text:
- 0: not mentioned.
- 1: mentioned in passing or alluded to.
- 2: a main argument of the text.

Rules:
- Score each frame independently. Note that several frames can score 2, and all can score 0, if necessary.
- A frame counts whether the text advances it itself or gives prominent space to sources who do; weigh how much of the text is devoted to it.
- The frames are listed in random order; the order carries no meaning.
- Return a JSON object with each score under its exact frame_id. Do not add explanations."""
ARTICLE_ANNOTATION = """You are an expert media analyst annotating how news articles frame one event. You will see a list of frames, each as [frame_id] name: description, and one article.

For EVERY frame, score how salient it is in the article:
- 0: not mentioned.
- 1: mentioned in passing or alluded to.
- 2: a main argument of the article.

Rules:
- Score each frame independently. Note that several frames can score 2, and all can score 0, if necessary.
- A frame counts whether the article advances it itself or gives prominent space to sources who do; weigh how much of the article is devoted to it.
- The frames are listed in random order; the order carries no meaning.
- Return each score under its exact frame_id."""


def messages(system, user):
    """Build the system/user message pair without changing prompt text."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def opaque_ids(count):
    """Create temporary article labels from their positions in the prompt."""
    labels = []
    used = set()
    for position in range(1, count + 1):
        # Position labels hide corpus identities without changing article content.
        number = int(hashlib.sha256(str(position).encode()).hexdigest()[:5], 16)
        while number in used:
            number = (number + 1) % 1048576
        used.add(number)
        labels.append(f"{number:05x}")
    return labels


def retrieval(pool, query, replicate):
    """Build a headline-selection prompt with a reproducible article permutation."""
    # Replicate fixes the headline permutation used in the original request.
    presented = pool.sample(frac=1, random_state=replicate)
    payload = [
        dict(article_id=label, headline=headline)
        for label, headline in zip(opaque_ids(len(pool)), presented.headline)
    ]
    return messages(RETRIEVAL + json.dumps(payload, ensure_ascii=False, indent=2), query)


def synthesis(articles, ids, query):
    """Build a synthesis prompt from the selected articles' titles and full bodies."""
    # Preserve retrieved order and full bodies; do not add publisher metadata.
    payload = [
        dict(title=articles.loc[article_id, "headline"], body=articles.loc[article_id, "body"])
        for article_id in ids
    ]
    if any(article["body"].strip() == "[REDACTED]" for article in payload):
        raise ValueError("Synthesis requires the original article bodies.")
    return messages(SYNTHESIS.format(articles=json.dumps(payload, ensure_ascii=False, indent=2)), query)


def annotation(event, frames, order, text, headline=None):
    """Use the article or answer template with the requested frame order."""
    if text.strip() == "[REDACTED]":
        raise ValueError("Annotation requires the original article or answer text.")
    frame_text = "\n".join(
        f"[{frame_id}] {frames[frame_id]['name']}: {frames[frame_id]['description']}"
        for frame_id in order
    )
    if headline is None:
        user = f'Event: {event}\n\nFrames:\n{frame_text}\n\nText:\n"""\n{text}\n"""'
        system = ANSWER_ANNOTATION
    else:
        user = f'Event: {event}\n\nFrames:\n{frame_text}\n\nHeadline: {headline}\n\nArticle:\n"""\n{text}\n"""'
        system = ARTICLE_ANNOTATION
    return messages(system, user)


# Original extraction, grouping and assignment templates.
EXTRACTION_SYSTEM_PROMPT = '''You are an expert media analyst. You read news articles about one event, {event}, and identify the media frame each article adopts on that event and the information it uses to build it.

Given an article, return:
1. media_frame: the MEDIA FRAME on {event}, i.e. the evaluative stance or opinion the article conveys on the event, whether stated outright or implied through framing, emphasis, and sourcing.
2. key_points: exactly 5 of the most salient pieces of information the article uses to build its frame on the event. For each, give:
   - info: the piece of information.
   - stance: whether the article "endorses", "refutes", or is "neutral" toward it.

Rules:
- Every text field must be ONE sentence of fewer than 20 words.
- Base everything strictly on the article text.
- Order key_points from most to least salient.'''

EXTRACTION_USER_TEMPLATE = '''Headline: {headline}

Article:
"""
{body}
"""'''

GROUPING_SYSTEM_PROMPT = '''You are an expert media analyst studying how news outlets covered one event. You will see the media frame of every article (the evaluative stance it conveys), one per line as [id] text.

Group these frames into between 5 and 12 categories by the POSITION they take: what they claim, or how they evaluate the event and its actors.

Rules:
- Group by position, not by sub-topic or wording. Two frames about different details (e.g. a court ruling vs. the original decision) belong together if they express the same position.
- Ignore framing boilerplate such as "The article conveys" or "The article frames".
- Put neutral, descriptive coverage that takes no position in one group named exactly "Neutral reporting".
- Categories must be mutually exclusive and together cover nearly every frame.
- name: at most 6 words, phrased as the position itself (e.g. "Officials overstepped their legal authority").
- description: one sentence saying what frames in this group have in common.
- Do not create an "Other" category; one is added automatically.'''

GROUPING_USER_TEMPLATE = '''Event: {event}

Frames:
{frames}'''

ASSIGNMENT_SYSTEM_PROMPT = '''You are an expert media analyst. You will see a list of frame categories and a list of frames, one per line as [id] text.

Assign every frame to exactly one category, based on the position it takes (not its sub-topic or wording). Use "Other" only if no category fits.

For each frame return its frame_id, the category name exactly as given, and a confidence: "high" if it is a clear example of the category, "medium" if it fits reasonably, "low" if it is a borderline fit.'''

ASSIGNMENT_USER_TEMPLATE = '''Event: {event}

Categories:
{groups}

Frames:
{frames}'''


def extraction_request(article):
    """Build the structured extraction request for one article and its event."""
    if article.body.strip() == "[REDACTED]":
        raise ValueError("Frame extraction requires the original article body.")
    system = EXTRACTION_SYSTEM_PROMPT.format(event=article.event_name)
    user = EXTRACTION_USER_TEMPLATE.format(headline=article.headline, body=article.body)
    return request(
        "frame_extraction", CONFIG["frame_extraction_model"], messages(system, user),
        settings={**CONFIG["frame_extraction_settings"], "schema_name": "ArticleAnalysis"},
    )


def lines(table):
    """Format extracted frames with event-wide positions that remain fixed across batches."""
    return "\n".join(f"[{row.position}] {row.extracted_frame}" for row in table.itertuples())


def grouping_request(event, table):
    """Build one grouping request from an event's ordered extracted frames."""
    user = GROUPING_USER_TEMPLATE.format(event=event, frames=lines(table))
    return request(
        "frame_grouping", CONFIG["frame_grouping_model"],
        messages(GROUPING_SYSTEM_PROMPT, user),
        settings={**CONFIG["frame_grouping_settings"], "schema_name": "FrameGroups"},
    )


def assignment_request(event, table, groups):
    """Build an assignment request for a frame batch, including the Other category."""
    group_text = "\n".join(f"- {group['name']}: {group['description']}" for group in groups)
    group_text += "\n- Other: fits none of the categories above."
    user = ASSIGNMENT_USER_TEMPLATE.format(event=event, groups=group_text, frames=lines(table))
    return request(
        "frame_assignment", CONFIG["frame_grouping_model"],
        messages(ASSIGNMENT_SYSTEM_PROMPT, user),
        settings={**CONFIG["frame_grouping_settings"], "schema_name": "FrameAssignments"},
    )
