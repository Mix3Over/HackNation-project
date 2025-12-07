export const moduleName = "@bluevoid/cli";

export type ExportsConfig = {
	inDir?: string;
	outDir?: string;
	moreFiles?: string[];
	moreExports?: Record<string, string>;
	moreTsPaths?: Record<string, string[]>;
	overrideModule?: string;
	overrideTypes?: string;
};
