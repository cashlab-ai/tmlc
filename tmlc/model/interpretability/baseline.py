from typing import List
import torch

def calculate_baseline(self, data: List[str], method: str, **kwargs) -> torch.Tensor:
    """
    Calculate the baseline input using the specified method.
    
    Args:
        data (List[str]): A list containing the input text.
        method (str): The method to use for calculating the baseline.
        **kwargs: Additional keyword arguments required by the selected method.
    
    Returns:
        torch.Tensor: The baseline input tensor.
    """

    def all_padding_tokens() -> torch.Tensor:
        """
        Create a baseline tensor filled with padding tokens.
        
        Returns:
            torch.Tensor: A tensor filled with padding tokens.
        """
        input_ids = self.tokenizer.encode(data[0], return_tensors="pt")
        padding_token_id = self.tokenizer.tokenizer.pad_token_id
        return torch.full(input_ids.shape, padding_token_id, dtype=torch.long)
    
    def average_embedding(negative_cases: List[str]) -> torch.Tensor:
        """
        Calculate the average embedding of a list of negative cases.
        
        Args:
            negative_cases (List[str]): A list of negative cases.
        
        Returns:
            torch.Tensor: The average embedding tensor.
        """
        embeddings = []
        for text in negative_cases:
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            embedding = self.model.backbone.embeddings(input_ids)
            embeddings.append(embedding)
        baseline = torch.mean(torch.cat(embeddings), dim=0)
        return baseline
    
    def random_baseline(n: int) -> torch.Tensor:
        """
        Calculate the average embedding of a random set of tokens.
        
        Args:
            n (int): The number of random tokens to generate.
        
        Returns:
            torch.Tensor: The average embedding tensor of random tokens.
        """
        random_text = []
        for _ in range(n):
            tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(
                torch.randint(0, len(self.tokenizer.tokenizer.vocab), (len(data[0]),))
            )
            random_text.append(self.tokenizer.tokenizer.convert_tokens_to_string(tokens))
        baseline = average_embedding(random_text)
        return baseline

    baseline_methods = {
        "all_padding_tokens": all_padding_tokens,
        "average_embedding": average_embedding,
        "random_baseline": random_baseline,
    }

    assert method in baseline_methods, f"Unsupported baseline method '{method}'."
    return baseline_methods[method](**kwargs)
