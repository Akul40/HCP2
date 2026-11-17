
import os
import json
import re
import time

from google import genai


# --------------------------------------------------
# 1. GEMINI CLIENT
# --------------------------------------------------

def get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key is not configured. "
            "Set GEMINI_API_KEY before starting the backend."
        )

    return genai.Client(api_key=api_key)


def get_model():

    return os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash"
    )


# --------------------------------------------------
# 2. GEMINI GENERATION WITH RETRIES AND FALLBACK
# --------------------------------------------------


def generate_with_fallback(client, prompt, config=None):

    primary_model = get_model()

    # Remove duplicate model names
    models = list(dict.fromkeys([
        primary_model,
        "gemini-3.7-flash",
        "gemini-3.6-flash"
    ]))

    errors = []

    for model in models:

        try:
            print(f"Trying Gemini model: {model}")

            request = {
                "model": model,
                "contents": prompt
            }

            if config is not None:
                request["config"] = config

            response = client.models.generate_content(**request)

            if response and response.text:
                print(f"Gemini response received from {model}")
                return response

            raise ValueError("Gemini returned an empty response.")

        except Exception as e:

            error_message = str(e).lower()
            errors.append(f"{model}: {str(e)}")

            temporary_error = any(
                word in error_message
                for word in [
                    "503",
                    "unavailable",
                    "high demand",
                    "500",
                    "504"
                ]
            )

            if temporary_error:
                print(f"{model} temporarily unavailable. Moving to next model.")
                continue

            raise

    raise RuntimeError(
        "All Gemini models failed:\n" + "\n".join(errors)
    )


# --------------------------------------------------
# 3. CLUSTERING ANALYSIS
# --------------------------------------------------

def analyze_clustering(results):

    client = get_gemini_client()

    actual_results = json.dumps(
        results,
        indent=2,
        default=str
    )

    prompt = f"""
You are an expert pharmaceutical business intelligence
and machine learning analyst.

Your task is to interpret actual Healthcare Professional
(HCP) segmentation results for a business audience.

You must base your analysis ONLY on the supplied data.

==================================================
ACTUAL CLUSTERING RESULTS
==================================================

{actual_results}


==================================================
REPORT REQUIREMENTS
==================================================

Prepare a professional report using the following sections.

1. EXECUTIVE SUMMARY

- Summarise the segmentation exercise.
- Identify the three clustering techniques evaluated.
- State which technique has the highest available
  silhouette score.
- Explain whether the highest score alone is enough
  to select a business solution.
- Mention important limitations.

2. TECHNIQUE-WISE ANALYSIS

A. K-Means

- Explain how it groups similar HCP records.
- Discuss its advantages and limitations.
- Interpret its actual silhouette score and cluster sizes.

B. Hierarchical Clustering

- Explain how it builds groups based on similarity.
- Discuss its advantages and limitations.
- Interpret its actual results.

C. DBSCAN

- Explain density-based grouping in simple language.
- Explain how it identifies noise or unusual observations.
- Discuss its actual cluster count and noise count.
- Explain the implications of the current DBSCAN parameters.

Do not invent performance measurements.

3. MODEL COMPARISON

Compare the three approaches on:

- Cluster separation.
- Interpretability.
- Handling unusual observations.
- Business suitability.
- Sensitivity to parameters.

Use the actual silhouette scores and cluster counts.

If K-Means and Hierarchical have the same partition,
explain that cluster numbering can differ even when
the underlying group membership is identical.

Use assignment_comparison only as evidence for this claim.

4. ELBOW METHOD AND CLUSTER COUNT

Interpret the elbow_analysis data.

Explain:

- What inertia represents.
- How inertia changes as K increases.
- Whether the elbow method suggests a possible cluster count.
- How silhouette scores across different K values
  provide additional evidence.

Do not claim that the suggested K is automatically optimal.

Explain that the business should also consider whether
the resulting segments are meaningful and actionable.

5. HCP SEGMENT INTERPRETATION

Use the actual cluster_profiles provided for each method.

For every available cluster:

- Report the number of HCPs.
- Describe prescription behaviour using the supplied
  numeric feature means and medians.
- Describe engagement using the supplied engagement data.
- Explain categorical distributions where available.
- Identify meaningful differences between clusters.

Do not invent cluster characteristics.

Do not assume that a higher prescription count means
higher future potential.

If the available data does not support a conclusion,
explicitly state that more information is required.

6. DBSCAN NOISE ANALYSIS

- Explain what noise observations mean.
- State the actual noise count.
- Identify HCP IDs only if they are explicitly present
  in the supplied assignments.
- Do not invent HCP IDs or describe noise records
  without evidence.
- Explain that noise does not automatically mean
  poor performance or low business value.

7. BUSINESS RECOMMENDATIONS

Provide practical, non-clinical recommendations.

Examples:

- Review differences in prescription behaviour.
- Examine engagement patterns across segments.
- Identify opportunities for further business analysis.
- Validate segments with domain experts.
- Monitor segment stability over time.

Do not make clinical or treatment recommendations.

Do not assume that a segment should automatically
receive a particular sales or medical engagement strategy.

8. LIMITATIONS AND NEXT STEPS

Discuss:

- Dataset size and representativeness.
- Selected features and feature weights.
- Sensitivity to clustering parameters.
- Need for cluster stability validation.
- Need for domain expert validation.
- Potential impact of missing or biased data.

==================================================
IMPORTANT RULES
==================================================

1. Never invent metrics, HCP IDs, cluster sizes or findings.

2. Distinguish actual findings from suggestions.

3. Do not recommend a technique using silhouette score alone.

4. Do not treat DBSCAN noise as a separate business segment.

5. Do not assume that different cluster labels mean
   different group membership.

6. Do not make clinical recommendations.

7. Use clear, concise, professional business language.

8. If information is missing, say so rather than guessing.

9. Interpret only the actual profiles supplied.

10. Do not claim that a model has been validated or
    deployed based only on these clustering results.

Use clear headings, short paragraphs and bullet points.
"""

    response = generate_with_fallback(
        client=client,
        prompt=prompt
    )

    return response.text


# --------------------------------------------------
# 4. NATURAL LANGUAGE FEATURE WEIGHT SUGGESTIONS
# --------------------------------------------------

def suggest_feature_adjustments(
    instruction,
    selected_features,
    current_weights
):

    client = get_gemini_client()

    available_features = list(selected_features)

    # Validate current weights
    weights = {}

    for feature in available_features:

        try:

            value = float(
                current_weights.get(feature, 1)
            )

            if not 0 <= value <= 5:
                value = 1

        except (ValueError, TypeError):

            value = 1

        weights[feature] = value

    prompt = f"""
You are an AI assistant helping a business user
configure HCP segmentation.

The user wants to modify the importance of selected
features used by a clustering algorithm.

USER INSTRUCTION:

{instruction}


AVAILABLE FEATURES:

{json.dumps(available_features)}


CURRENT FEATURE WEIGHTS:

{json.dumps(weights)}


YOUR TASK:

Translate the user's instruction into proposed feature weights.

RULES:

1. Only modify features from AVAILABLE FEATURES.

2. Preserve the current weight of features that the
   instruction does not mention.

3. Use numerical weights between 0 and 5.

4. A weight of:
   - 0 means exclude the feature.
   - 1 means normal importance.
   - Above 1 means increased importance.
   - Between 0 and 1 means reduced importance.

5. Do not invent new features.

6. Do not change clustering algorithms or cluster counts.

7. If the instruction is unclear, preserve all current
   weights and explain what needs clarification.

8. Do not claim that a proposed weight will guarantee
   a particular clustering outcome.

9. Return every available feature in proposed_weights.

Return ONLY valid JSON in this exact structure:

{{
    "proposed_weights": {{
        "feature_name": 1.0
    }},
    "explanation": "Explain the proposed changes in simple business language."
}}
"""

    response = generate_with_fallback(
        client=client,
        prompt=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    # Parse Gemini JSON response
    try:

        suggestion = json.loads(response.text)

    except json.JSONDecodeError:

        cleaned = re.sub(
            r"```(?:json)?|```",
            "",
            response.text
        ).strip()

        try:

            suggestion = json.loads(cleaned)

        except json.JSONDecodeError:

            raise ValueError(
                "Gemini returned an invalid JSON adjustment response."
            )

    proposed_weights = suggestion.get(
        "proposed_weights",
        {}
    )

    if not isinstance(proposed_weights, dict):
        proposed_weights = {}

    # Validate proposed weights
    validated_weights = {}

    for feature in available_features:

        proposed_value = proposed_weights.get(
            feature,
            weights[feature]
        )

        try:

            proposed_value = float(proposed_value)

        except (ValueError, TypeError):

            proposed_value = weights[feature]

        # Reject NaN, infinity and out-of-range values
        if (
            not 0 <= proposed_value <= 5
            or not (
                proposed_value == proposed_value
                and abs(proposed_value) != float("inf")
            )
        ):

            proposed_value = weights[feature]

        validated_weights[feature] = round(
            proposed_value,
            2
        )

    explanation = suggestion.get(
        "explanation",
        "Feature weights have been proposed based on your instruction."
    )

    if not isinstance(explanation, str):
        explanation = str(explanation)

    return {

        "proposed_weights": validated_weights,

        "explanation": explanation

    }
