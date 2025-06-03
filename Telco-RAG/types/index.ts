export enum OpenAIModel {
  DAVINCI_TURBO = "gpt-4o-mini-turbo"
}

export type Source = {
  url: string;
  text: string;
};

export type SearchQuery = {
  query: string;
  sourceLinks: string[];
};
