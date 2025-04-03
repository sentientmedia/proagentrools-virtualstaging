from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def predict(
        self,
        prompt: str = Input(description="Prompt to generate something cool"),
    ) -> str:
        return f"You said: {prompt}"
