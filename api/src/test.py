def calculate_model_perplexity(self, datasets=['gsm8k'], seqlen=32): # , use_cuda_graph=False, use_flash_attn=False
    model = self.model
    model_str = self.model_name
    acc_loss = 0.0
    total_samples = 0

    for dataset in datasets:
        input_tok = gptq_data_utils.get_test_tokens(dataset, seed=0, seqlen=seqlen, model=model_str)
        total_length = input_tok.size(0)
        nsamples = total_length // seqlen
        rest = total_length % seqlen

        if rest != 0:
        # if the last part of the data is not complete, we cut it off
            input_tok = input_tok[:-rest]

        input_tok = input_tok.view(-1, seqlen)  # reshape the tensor
        total_samples += nsamples

        #if not use_cuda_graph:
        #    model.reset()

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(input, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

    avg_loss = acc_loss / total_samples
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl